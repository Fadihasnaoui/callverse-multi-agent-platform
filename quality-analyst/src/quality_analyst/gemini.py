"""Official Google GenAI SDK, fixed endpoint, bounded requests and no retries."""
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4
import json,logging,os,time
from quality_analyst.ollama import AdapterError
from quality_analyst.release_config import settings

ENDPOINT='https://generativelanguage.googleapis.com'
MAX_RESPONSE_BYTES=131072

def read_api_key():
    """Credentials never become settings fields, request packets, or session data."""
    path=settings().gemini_key_file
    try:
        if path.is_file():
            with path.open('rb') as stream:raw=stream.read(4097)
            if len(raw)>4096:raise AdapterError('GEMINI_KEY_FILE_INVALID')
            key=raw.decode('utf-8-sig').strip()
        else:key=''
        if not key:key=os.environ.get('GEMINI_API_KEY','').strip()
        if not key:raise AdapterError('GEMINI_KEY_NOT_CONFIGURED')
        if not 20<=len(key)<=4096 or any(c.isspace() or ord(c)<32 for c in key):
            raise AdapterError('GEMINI_KEY_FILE_INVALID')
        return key
    except (OSError,UnicodeError):raise AdapterError('GEMINI_KEY_FILE_UNREADABLE') from None

def reserve_call(stage,session_id):
    """Called under the provider session lock, before each attempted POST.

    Reservations remain consumed after any error/timeout. UTC days, no billing
    discovery or free-tier guarantee; the operator must use an unbilled project.
    """
    s=settings();now=datetime.now(timezone.utc)
    folder=s.data_dir/'gemini_calls'/now.strftime('%Y-%m-%d');folder.mkdir(parents=True,exist_ok=True)
    if sum(1 for _ in folder.glob('*.json'))>=s.gemini_max_calls_per_day:
        raise AdapterError('GEMINI_LOCAL_DAILY_LIMIT_REACHED')
    record={'reserved_at':now.isoformat(),'session_id':session_id,'stage':stage,
        'model':s.gemini_model,'maximum_generation_attempts':1,'automatic_retries':0}
    with (folder/(uuid4().hex+'.json')).open('x',encoding='utf-8') as f:json.dump(record,f)

def safe_error(exc):
    """Never stringify vendor/network exceptions: they may carry headers or keys."""
    if isinstance(exc,AdapterError):return exc
    code=getattr(exc,'code',None)
    if code in {401,403}:return AdapterError('GEMINI_AUTH_OR_ACCESS_REFUSED')
    if code==404:return AdapterError('GEMINI_MODEL_UNAVAILABLE')
    if code==429:return AdapterError('GEMINI_QUOTA_EXHAUSTED_NO_RETRY')
    if code==400:return AdapterError('GEMINI_REQUEST_OR_SCHEMA_REFUSED')
    if code in {408,504}:return AdapterError('GEMINI_TIMEOUT_NO_RETRY')
    if type(exc).__name__ in {'TimeoutException','ReadTimeout','ConnectTimeout','WriteTimeout','PoolTimeout'}:
        return AdapterError('GEMINI_TIMEOUT_NO_RETRY')
    return AdapterError('GEMINI_UNAVAILABLE_NO_RETRY')

class GeminiClient:
    def __init__(self,*,transport=None):
        self.generation_attempts=0;self.started=time.monotonic();self.identity=None
        self._transport=transport

    def _open(self):
        for name in ('google','httpx','httpcore'):logging.getLogger(name).setLevel(logging.CRITICAL)
        from google import genai
        from google.genai import types
        import httpx
        key=read_api_key();s=settings();seen=set();underlying=self._transport or httpx.HTTPTransport(retries=0)
        model=s.gemini_model
        allowed={('GET','/v1beta/models/'+model),('POST','/v1beta/models/'+model+':generateContent')}
        class BoundedStream(httpx.SyncByteStream):
            def __init__(self,stream):self.stream=stream
            def __iter__(self):
                size=0
                for chunk in self.stream:
                    size+=len(chunk)
                    if size>MAX_RESPONSE_BYTES:raise AdapterError('GEMINI_RESPONSE_TOO_LARGE')
                    yield chunk
            def close(self):self.stream.close()
        class SingleAttemptTransport(httpx.BaseTransport):
            def handle_request(self,request):
                route=(request.method,request.url.path)
                if request.url.scheme!='https' or request.url.host!='generativelanguage.googleapis.com' or request.url.port not in {None,443} or request.url.query or route not in allowed:
                    raise AdapterError('GEMINI_ENDPOINT_REFUSED')
                if route in seen:raise AdapterError('GEMINI_SECOND_HTTP_ATTEMPT_REFUSED')
                seen.add(route)
                response=underlying.handle_request(request)
                response.stream=BoundedStream(response.stream)
                return response
            def close(self):underlying.close()
        self.http=httpx.Client(transport=SingleAttemptTransport(),trust_env=False,
            follow_redirects=False,timeout=s.gemini_request_timeout)
        self.client=genai.Client(vertexai=False,api_key=key,http_options=types.HttpOptions(
            base_url=ENDPOINT,api_version='v1beta',timeout=s.gemini_request_timeout*1000,
            retry_options=types.HttpRetryOptions(attempts=1),httpx_client=self.http))
        return types

    def inspect_model(self):
        from google.genai import types
        if not hasattr(self,'client'):self._open()
        remaining=settings().gemini_request_timeout-(time.monotonic()-self.started)
        if remaining<=0:raise AdapterError('GEMINI_TIMEOUT_NO_RETRY')
        try:
            model=self.client.models.get(model=settings().gemini_model,config=types.GetModelConfig(
                http_options=types.HttpOptions(timeout=int(min(10,remaining)*1000))))
            if model.name not in {settings().gemini_model,'models/'+settings().gemini_model}:
                raise AdapterError('GEMINI_MODEL_IDENTITY_MISMATCH')
            if 'generateContent' not in (model.supported_actions or []):
                raise AdapterError('GEMINI_GENERATION_NOT_SUPPORTED')
            self.identity={'name':model.name,'supported_actions':list(model.supported_actions or []),
                'input_token_limit':model.input_token_limit,'output_token_limit':model.output_token_limit}
            return self.identity
        except Exception as exc:raise safe_error(exc) from None

    def generate(self,request,*,reserve):
        from google.genai import types
        if self.generation_attempts:raise AdapterError('SECOND_GENERATION_REFUSED')
        if self.identity is None:self.inspect_model()
        remaining=settings().gemini_request_timeout-(time.monotonic()-self.started)
        if remaining<=0:raise AdapterError('GEMINI_TIMEOUT_NO_RETRY')
        if self.identity.get('output_token_limit') and self.identity['output_token_limit']<request['max_output_tokens']:
            raise AdapterError('GEMINI_MODEL_OUTPUT_LIMIT_TOO_SMALL')
        config=types.GenerateContentConfig(temperature=0,seed=42,candidate_count=1,
            max_output_tokens=request['max_output_tokens'],response_mime_type='application/json',
            response_json_schema=request['format'],tools=[],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            http_options=types.HttpOptions(timeout=max(1,int(remaining*1000))))
        if settings().gemini_model.startswith('gemini-2.5-flash'):
            config.thinking_config=types.ThinkingConfig(thinking_budget=0)
        try:
            reserve();self.generation_attempts=1;started=time.monotonic()
            response=self.client.models.generate_content(model=settings().gemini_model,
                contents=request['prompt'],config=config)
            if time.monotonic()-self.started>settings().gemini_request_timeout:
                raise AdapterError('GEMINI_TIMEOUT_NO_RETRY')
            candidates=response.candidates or []
            if len(candidates)!=1:raise AdapterError('GEMINI_RESPONSE_BLOCKED_OR_EMPTY')
            candidate=candidates[0]
            reason=getattr(candidate.finish_reason,'value',candidate.finish_reason)
            if reason!='STOP':raise AdapterError('GEMINI_RESPONSE_INCOMPLETE_OR_BLOCKED')
            parts=candidate.content.parts if candidate.content else []
            if not parts or any(p.function_call or p.executable_code or p.code_execution_result or p.inline_data for p in parts):
                raise AdapterError('GEMINI_NON_TEXT_RESPONSE_REFUSED')
            text=''.join(p.text or '' for p in parts if not p.thought)
            if not text or len(text.encode('utf-8'))>MAX_RESPONSE_BYTES:raise AdapterError('GEMINI_RESPONSE_TOO_LARGE_OR_EMPTY')
            usage=response.usage_metadata
            if usage and usage.candidates_token_count is not None and usage.candidates_token_count>request['max_output_tokens']:
                raise AdapterError('GEMINI_TOKEN_BUDGET_EXCEEDED')
            return {'provider':'gemini','model':settings().gemini_model,'model_version':response.model_version,
                'response':text,'finish_reason':reason,'done':True,'generation_seconds':time.monotonic()-started,
                'tokens':{'input':usage.prompt_token_count if usage else None,
                    'output':usage.candidates_token_count if usage else None,
                    'thinking':usage.thoughts_token_count if usage else None}}
        except Exception as exc:raise safe_error(exc) from None

    def close(self):
        if hasattr(self,'client'):self.client.close()
        if hasattr(self,'http'):self.http.close()
