"""Cloud stage with shared business verification, no Ollama probes or fallback."""
from datetime import datetime,timezone
from pathlib import Path
import time
import psutil
from quality_analyst import ozone_quality_v02 as q,interactive
from quality_analyst.gemini import GeminiClient,reserve_call,safe_error
from quality_analyst.ollama import GenerationConfig,INSTRUCT_MODEL,AdapterError,sha256
from quality_analyst.release_config import settings

def prepare(c,*,stage='context',context=None,retrieval=None):
    # Fixed business budget independent of the local CPU settings. Same normalized
    # prompt, schema, comparisons and citations as the default local engine.
    config=GenerationConfig(num_ctx=8192,num_predict=2048,think=False,model=INSTRUCT_MODEL)
    local,budget,checks=q.prepare(c,stage=stage,context=context,retrieval=retrieval,generation_config=config)
    return {'provider':'gemini','model':settings().gemini_model,'prompt':local['prompt'],
        'format':local['format'],'max_output_tokens':2048},budget,checks

def execute(c,directory,*,stage='context',context=None,retrieval=None,cloud_consent=False,free_tier_confirmed=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);started=time.monotonic();client=None
    process=psutil.Process()
    result={'engine':q.VERSION,'provider':'gemini','model':settings().gemini_model,'model_digest':None,
        'stage':stage,'status':'blocked','checked':None,'errors':[],'generation_calls':0,'automatic_retries':0,
        'started_at':datetime.now(timezone.utc).isoformat(),'semantic_validated':False,'historical_categories_required':False,
        'input_sha256':sha256(q.compact(c.model_dump(mode='json')).encode()),
        'consent_recorded':cloud_consent is True,'unbilled_project_confirmed_by_operator':free_tier_confirmed is True,
        'process_rss_before_bytes':process.memory_info().rss}
    try:
        if cloud_consent is not True:raise AdapterError('GEMINI_CONSENT_REQUIRED')
        if free_tier_confirmed is not True:raise AdapterError('GEMINI_UNBILLED_PROJECT_CONFIRMATION_REQUIRED')
        request,budget,checks=prepare(c,stage=stage,context=context,retrieval=retrieval)
        interactive.ensure_no_secrets(request);result['admission']=budget
        interactive.write_json(directory/'request.json',request);interactive.write_json(directory/'comparisons.json',checks)
        client=GeminiClient();identity=client.inspect_model();interactive.write_json(directory/'identity.json',identity)
        envelope=client.generate(request,reserve=lambda:reserve_call(stage,directory.parent.name))
        interactive.ensure_no_secrets(envelope)
        interactive.write_json(directory/'raw_response.json',envelope)
        with (directory/'raw_response.txt').open('x',encoding='utf-8',newline='') as f:f.write(envelope['response'])
        checked=q.validate_output(envelope['response'],c,stage=stage,comparisons=checks)
        result.update(status=checked['status'],checked=checked,tokens=envelope['tokens'],
            generation_seconds=envelope['generation_seconds'],model_version=envelope['model_version'])
    except Exception as exc:
        error=safe_error(exc)
        result.update(status='failed' if client and client.generation_attempts else 'blocked',errors=[error.code])
    finally:
        if client:client.close()
        result.update(generation_calls=client.generation_attempts if client else 0,wall_seconds=time.monotonic()-started,
            completed_at=datetime.now(timezone.utc).isoformat(),process_rss_after_bytes=process.memory_info().rss)
        interactive.write_json(directory/'result.json',result)
    return result
