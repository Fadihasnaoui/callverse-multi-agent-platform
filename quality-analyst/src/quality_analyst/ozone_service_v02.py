"""Immutable provider-bound sessions, independent locks, no automatic replay."""
from pathlib import Path
from datetime import datetime,timezone
from uuid import uuid4
import os,subprocess,time,sys
from quality_analyst.release_config import settings
from quality_analyst.interactive import write_json,ensure_no_secrets
from quality_analyst.ollama import compact,sha256
from quality_analyst.contracts import strict_json_loads

ROOT=Path(__file__).resolve().parents[2];SESSIONS=settings().data_dir/'ozone_v02_sessions'

def run(c,*,stage='context',context=None,retrieval=None,session_root=SESSIONS,
        provider=None,cloud_consent=False,free_tier_confirmed=False):
    if stage not in {'context','analysis'}:raise ValueError('STAGE_REFUSED')
    provider=provider or settings().ai_provider
    if provider not in {'ollama','gemini'}:raise ValueError('AI_PROVIDER_REFUSED')
    ensure_no_secrets(c.model_dump(mode='json'))
    directory=Path(session_root)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_')+uuid4().hex[:8])
    directory.mkdir(parents=True,exist_ok=False)
    model=settings().gemini_model if provider=='gemini' else settings().ollama_model
    payload={'conversation':c.model_dump(mode='json'),'stage':stage,'context':context,
        'retrieval':retrieval.model_dump(mode='json') if retrieval else None,'provider':provider,'model':model,
        'cloud_consent':cloud_consent is True,'free_tier_confirmed':free_tier_confirmed is True}
    ensure_no_secrets(payload);write_json(directory/'input.json',payload)
    metadata={'session_id':directory.name,'engine':'qa.ozone_live_quality/0.2.0','stage':stage,
        'provider':provider,'model':model,'started_at':datetime.now(timezone.utc).isoformat(),
        'input_sha256':sha256(compact(c.model_dump(mode='json')).encode()),'packet_sha256':sha256(compact(payload).encode()),
        'dataset_role':'interactive_diagnostic_only','automatic_training_export':False,
        'semantic_validated':False,'maximum_generation_calls':1,'automatic_retries':0,
        'cloud_consent_recorded':cloud_consent is True if provider=='gemini' else False,
        'unbilled_project_confirmed_by_operator':free_tier_confirmed is True if provider=='gemini' else False}
    write_json(directory/'reservation.json',metadata)
    lock=settings().data_dir/('.ollama.lock' if provider=='ollama' else '.gemini.lock')
    lock.parent.mkdir(parents=True,exist_ok=True);acquired=False;uncertain=False;begin=time.monotonic()
    outcome={'status':'blocked','generation_calls':0,'errors':[],'checked':None}
    try:
        if provider=='gemini' and cloud_consent is not True:
            outcome['errors']=['GEMINI_CONSENT_REQUIRED']
        elif provider=='gemini' and free_tier_confirmed is not True:
            outcome['errors']=['GEMINI_UNBILLED_PROJECT_CONFIRMATION_REQUIRED']
        else:
            with lock.open('x',encoding='utf-8') as f:f.write(compact({'session_id':directory.name,'provider':provider,'stage':stage,'owner_pid':os.getpid()}))
            acquired=True
            command=[sys.executable,'-B',str(ROOT/'scripts/ozone_worker_v02.py'),'--session',str(directory)]
            timeout=settings().generation_timeout+45 if provider=='ollama' else settings().gemini_request_timeout+30
            completed=subprocess.run(command,cwd=ROOT,capture_output=True,timeout=timeout,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            path=directory/'runtime/result.json'
            if path.exists():outcome=strict_json_loads(path.read_bytes())
            else:
                uncertain=True;outcome.update(status='failed',errors=['WORKER_FAILED_NO_RESULT'])
            if completed.returncode:
                uncertain=True;outcome.update(status='failed',errors=outcome['errors']+['WORKER_NONZERO_EXIT'])
    except FileExistsError:outcome['errors']=['ANALYSIS_ALREADY_IN_PROGRESS_NO_RETRY']
    except subprocess.TimeoutExpired:uncertain=True;outcome.update(status='failed',generation_calls=None,errors=['WORKER_TIMEOUT_NO_RETRY'])
    except (ValueError,OSError):outcome.update(status='failed',errors=['LOCAL_EXECUTION_ERROR'])
    finally:
        if acquired and outcome.get('generation_calls'):
            if provider=='ollama' and not outcome.get('server_idle_confirmed',False):uncertain=True
            if provider=='gemini' and 'GEMINI_TIMEOUT_NO_RETRY' in outcome.get('errors',[]):uncertain=True
        if acquired and not uncertain:lock.unlink()
        result={**metadata,**outcome,'wall_seconds':time.monotonic()-begin,'worker_state_uncertain':uncertain}
        write_json(directory/'result.json',result)
    return result

def history():
    rows=[]
    for p in sorted(SESSIONS.glob('*/result.json'),reverse=True):
        try:rows.append(strict_json_loads(p.read_bytes()))
        except (OSError,ValueError):pass
    return rows
