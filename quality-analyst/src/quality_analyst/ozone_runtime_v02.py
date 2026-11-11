"""Same guarded localhost transport, memory checks and pinned identity as V."""
from pathlib import Path
from datetime import datetime,timezone
import json,time
from quality_analyst import ozone_quality_v02 as q,interactive
from quality_analyst.evidence_guided_runtime import GuardedClient
from quality_analyst.release_hardware import preflight,confirm_idle
from quality_analyst.ollama import AdapterError,INSTRUCT_PROFILE,INSTRUCT_DIGEST,require_generation_mode,sha256

class Client(GuardedClient):
    def __init__(self,payload,stage):super().__init__();self.expected=payload;self.stage=stage
    def generate(self,payload,show):
        require_generation_mode(INSTRUCT_PROFILE,show)
        if self.generation_attempts:raise AdapterError('SECOND_GENERATION_REFUSED')
        if payload!=self.expected:raise AdapterError('CONFIGURATION_REFUSED')
        q.admission(payload,q.CONTEXT if self.stage=='context' else q.ANALYSIS)
        self.generation_attempts=1;return self.guarded_post(payload)

def execute(c,directory,*,stage='context',context=None,retrieval=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);started=time.monotonic();client=None
    result={'engine':q.VERSION,'stage':stage,'status':'blocked','checked':None,'errors':[],'generation_calls':0,
        'automatic_retries':0,'model_digest':INSTRUCT_DIGEST,'started_at':datetime.now(timezone.utc).isoformat(),
        'semantic_validated':False,'input_sha256':sha256(q.compact(c.model_dump(mode='json')).encode()),
        'historical_categories_required':False}
    try:
        payload,budget,checks=q.prepare(c,stage=stage,context=context,retrieval=retrieval)
        interactive.ensure_no_secrets(payload);result['admission']=budget;client=Client(payload,stage)
        interactive.write_json(directory/'request.json',payload);interactive.write_json(directory/'comparisons.json',checks)
        wire=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
        with (directory/'request_wire.json').open('xb') as f:f.write(wire)
        result['wire_sha256']=sha256(wire)
        snap,errors,_=preflight('qwen');result.update(resources_before=interactive.resources(snap),errors=errors)
        if not errors:
            show=client.inspect_model();interactive.write_json(directory/'identity.json',show)
            interactive.write_json(directory/'reservation.json',{'maximum_generations':1,'wire_sha256':sha256(wire),'automatic_retries':0})
            with interactive.hardware.Sampler() as monitor:
                try:
                    begin=time.monotonic();envelope=client.generate(payload,show);result['generation_seconds']=time.monotonic()-begin
                    interactive.ensure_no_secrets(envelope);interactive.write_json(directory/'raw_response.json',envelope)
                    with (directory/'raw_response.txt').open('x',encoding='utf-8',newline='') as f:f.write(envelope.get('response',''))
                    checked=q.validate(envelope,c,stage=stage,comparisons=checks)
                    result.update(status=checked['status'],checked=checked,tokens={k:envelope.get(k) for k in ['prompt_eval_count','eval_count','done_reason']})
                finally:result['resource_samples']=monitor.samples
            state=interactive.hardware.probes.probe_api('ps');result['model_state_after']=state
            if state.get('status')!='ok' or state.get('data',{}).get('models')!=[]:result['errors'].append('MODEL_NOT_CONFIRMED_UNLOADED')
    except AdapterError as exc:result.update(status='failed' if client and client.generation_attempts else 'blocked',errors=[exc.code])
    except (ValueError,OSError,KeyError,TypeError) as exc:result.update(status='failed',errors=[type(exc).__name__])
    finally:
        if client and client.generation_attempts:
            state,idle=confirm_idle()
            result.update(model_state_after=state,server_idle_confirmed=idle)
            if not idle and 'MODEL_NOT_CONFIRMED_UNLOADED' not in result['errors']:
                result['errors'].append('MODEL_NOT_CONFIRMED_UNLOADED')
        result.update(generation_calls=client.generation_attempts if client else 0,
            guard_samples=client.guard_samples if client else [],wall_seconds=time.monotonic()-started,
            completed_at=datetime.now(timezone.utc).isoformat())
        interactive.write_json(directory/'result.json',result)
    return result
