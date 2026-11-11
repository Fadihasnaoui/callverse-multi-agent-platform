from __future__ import annotations
from dataclasses import dataclass
import hashlib, http.client, json, socket, threading, time
from quality_analyst.contracts import strict_json_loads
from quality_analyst.release_config import settings, ollama_connection
BASE_URL = settings().ollama_base_url
MODEL = 'qwen3.5:4b'
INSTRUCT_MODEL = settings().ollama_model
# Exact local identity used by the Ozone Customer Advisor. The former release
# required a different Qwen 3 Instruct build and would refuse the Advisor model.
INSTRUCT_PROFILE = 'advisor_qwen35'
QUOTES_PROFILE = 'advisor_qwen35_quotes/0.1.0'
INSTRUCT_DIGEST = 'd8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04'
MAX_RESPONSE_BYTES = 128 * 1024
CONNECT_TIMEOUT = 5.0
TOTAL_TIMEOUT = 180.0

@dataclass(frozen=True, slots=True)
class GenerationConfig:
    num_ctx: int = 4096
    num_predict: int = 3072
    context_reserve: int = 256
    temperature: int = 0
    seed: int = 42
    think: bool = True
    stream: bool = False
    keep_alive: int = 0
    model: str = MODEL

    def __post_init__(self):
        integers = (self.num_ctx, self.num_predict, self.context_reserve, self.temperature, self.seed, self.keep_alive)
        if any((type(v) is not int for v in integers)) or type(self.think) is not bool or type(self.stream) is not bool or (self.num_ctx <= 0) or (self.num_predict <= 0) or (self.context_reserve < 0) or (self.input_token_budget <= 0) or (self.model not in (MODEL, INSTRUCT_MODEL)):
            raise ValueError('Invalid generation configuration')

    @property
    def input_token_budget(self):
        return self.num_ctx - self.num_predict - self.context_reserve

    def options(self):
        return {'num_ctx': self.num_ctx, 'num_predict': self.num_predict, 'temperature': self.temperature, 'seed': self.seed, **({'num_gpu': 0} if settings().hardware_mode == 'cpu' else {})}

    def settings(self):
        return {'model': self.model, 'think': self.think, 'stream': self.stream, 'keep_alive': self.keep_alive, 'options': self.options()}
CONFIG = GenerationConfig()
NO_THINK_PROFILE = 'no_think_probe'
NO_THINK_CONFIG = GenerationConfig(num_predict=1024)
INSTRUCT_CONFIG = GenerationConfig(num_predict=1024, think=False, model=INSTRUCT_MODEL)

def profile_config(profile):
    """Profils fermés ; les profils historiques gardent exactement leurs options."""
    if profile == 'retry2':
        return CONFIG
    if profile == NO_THINK_PROFILE:
        return NO_THINK_CONFIG
    if profile in (INSTRUCT_PROFILE, QUOTES_PROFILE):
        return INSTRUCT_CONFIG
    raise AdapterError('EXPERIMENT_PROFILE_REFUSED')

class AdapterError(Exception):

    def __init__(self, code: str, *, http_status: int | None=None):
        super().__init__(code)
        self.code = code
        self.http_status = http_status

def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def sha256(raw):
    return hashlib.sha256(raw).hexdigest()

def _thinking_metadata(show):
    if not isinstance(show, dict):
        raise AdapterError('INVALID_COMPATIBILITY_METADATA')
    controls, capabilities = (show.get('thinking'), show.get('capabilities'))
    if not isinstance(controls, dict) or not isinstance(capabilities, list) or (not all((isinstance(value, str) for value in capabilities))):
        raise AdapterError('INVALID_COMPATIBILITY_METADATA')
    values, default = (controls.get('values'), controls.get('default'))
    if not isinstance(values, list) or not values or (not all((type(value) is bool for value in values))) or (type(default) is not bool) or (not any((value is default for value in values))):
        raise AdapterError('INVALID_COMPATIBILITY_METADATA')
    return (controls, capabilities)

def require_think_true(show):
    controls, capabilities = _thinking_metadata(show)
    if not any((value is True for value in controls['values'])):
        raise AdapterError('THINK_TRUE_UNSUPPORTED')
    if 'completion' not in capabilities:
        raise AdapterError('COMPLETION_UNSUPPORTED')

def require_generation_mode(profile, show):
    if profile not in (INSTRUCT_PROFILE, QUOTES_PROFILE):
        profile_config(profile)
        return require_think_true(show)
    if not isinstance(show, dict) or show.get('advisor_model_verified') is not True or (not isinstance(show.get('capabilities'), list)) or (not all((isinstance(v, str) for v in show['capabilities']))):
        raise AdapterError('INSTRUCT_METADATA_MISMATCH')
    if 'completion' not in show['capabilities']:
        raise AdapterError('COMPLETION_UNSUPPORTED')
    if show.get('thinking') is not None:
        controls, _ = _thinking_metadata(show)
        if controls['values'] != [False] or controls['default'] is not False:
            raise AdapterError('NATIVE_NON_THINKING_REQUIRED')

class LocalOllama:
    """Transport direct sans proxy, redirection, retries ou téléchargement."""

    def __init__(self, base_url=BASE_URL, *, profile='retry2'):
        if base_url != BASE_URL:
            raise AdapterError('DESTINATION_REFUSED')
        self.generation_attempts = 0
        self.profile = profile
        self.config = profile_config(profile)

    def _post(self, endpoint, payload, total_timeout):
        if endpoint not in ('/api/show', '/api/generate'):
            raise AdapterError('ENDPOINT_REFUSED')
        connection = ollama_connection(CONNECT_TIMEOUT)
        expired = threading.Event()
        active_socket = None

        def interrupt():
            expired.set()
            if active_socket is not None:
                try:
                    active_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()
        timer = threading.Timer(total_timeout, interrupt)
        timer.daemon = True
        timer.start()
        started = time.monotonic()
        try:
            connection.connect()
            active_socket = connection.sock
            active_socket.settimeout(max(0.001, total_timeout - (time.monotonic() - started)))
            connection.request('POST', endpoint, body=compact(payload).encode('utf-8'), headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise AdapterError('REDIRECT_REFUSED', http_status=response.status)
            if response.status != 200:
                raise AdapterError('HTTP_ERROR', http_status=response.status)
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if expired.is_set() or time.monotonic() - started > total_timeout:
                raise AdapterError('INFERENCE_TIMEOUT')
            if len(raw) > MAX_RESPONSE_BYTES:
                raise AdapterError('RESPONSE_TOO_LARGE')
            try:
                data = strict_json_loads(raw)
                if not isinstance(data, dict):
                    raise ValueError
            except (ValueError, RecursionError):
                raise AdapterError('INVALID_ENVELOPE_JSON') from None
            if endpoint == '/api/generate':
                data.pop('thinking', None)
            return data
        except AdapterError:
            raise
        except (TimeoutError, socket.timeout):
            raise AdapterError('INFERENCE_TIMEOUT') from None
        except (OSError, http.client.HTTPException):
            code = 'INFERENCE_TIMEOUT' if expired.is_set() else 'SERVER_UNAVAILABLE'
            raise AdapterError(code) from None
        finally:
            timer.cancel()
            connection.close()

    def inspect_model(self):
        data = self._post('/api/show', {'model': self.config.model}, 5.0)
        if self.profile in (INSTRUCT_PROFILE, QUOTES_PROFILE):
            info, details = (data.get('model_info'), data.get('details'))
            capabilities = data.get('capabilities')
            if not isinstance(info, dict) or not isinstance(details, dict) or info.get('general.architecture') != 'qwen35' or details.get('quantization_level') != 'Q4_K_M' or details.get('parameter_size') != '4.2B' or (not isinstance(capabilities, list)) or (not all((isinstance(v, str) for v in capabilities))):
                raise AdapterError('INSTRUCT_METADATA_MISMATCH')
            controls = None
            if data.get('thinking') is not None:
                thinking, _ = _thinking_metadata(data)
                controls = {'values': list(thinking['values']), 'default': thinking['default']}
            result = {'capabilities': [v for v in capabilities if v in ('completion', 'tools', 'thinking')], 'thinking': controls, 'advisor_model_verified': True, 'non_thinking_basis': 'pinned_advisor_qwen35_metadata'}
            require_generation_mode(self.profile, result)
            return result
        thinking, capabilities = _thinking_metadata(data)
        return {'capabilities': [v for v in capabilities if v in ('completion', 'tools', 'thinking')], 'thinking': {'values': list(thinking['values']), 'default': thinking['default']}}
