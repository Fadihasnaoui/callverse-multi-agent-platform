"""Offline configuration by default. A real probe requires two explicit flags."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quality_analyst.release_config import settings,configure_logging
from quality_analyst.gemini import GeminiClient,read_api_key,reserve_call,safe_error
from quality_analyst.ollama import AdapterError

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--allow-cloud-send',action='store_true')
    parser.add_argument('--confirm-unbilled-project',action='store_true')
    args=parser.parse_args();configure_logging();s=settings()
    result={'provider':'gemini','model':s.gemini_model,'network_calls':0,'generation_calls':0,
        'automatic_retries':0,'key_value_omitted':True,'errors':[]}
    try:read_api_key();result['key_configured']=True
    except AdapterError as exc:result.update(key_configured=False,errors=[exc.code])
    if not args.probe:
        print(json.dumps(result));return 0 if result['key_configured'] else 1
    if not args.allow_cloud_send or not args.confirm_unbilled_project:
        result['errors']=['GEMINI_EXPLICIT_PROBE_CONSENT_REQUIRED'];print(json.dumps(result));return 2
    lock=s.data_dir/'.gemini.lock';lock.parent.mkdir(parents=True,exist_ok=True);acquired=False;client=None
    started=time.monotonic();folder=s.data_dir/'gemini_probe'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    try:
        with lock.open('x',encoding='utf-8') as f:f.write(json.dumps({'stage':'probe','started_at':datetime.now(timezone.utc).isoformat()}))
        acquired=True;client=GeminiClient();result['identity']=client.inspect_model()
        request={'prompt':'Return only JSON with ok equal to true.','format':{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False},'max_output_tokens':32}
        result['response']=client.generate(request,reserve=lambda:reserve_call('probe',folder.name))
        if json.loads(result['response']['response'])!={'ok':True}:raise AdapterError('GEMINI_PROBE_JSON_INVALID')
        result['errors']=[]
    except FileExistsError:result['errors']=['ANALYSIS_ALREADY_IN_PROGRESS_NO_RETRY']
    except Exception as exc:result['errors']=[safe_error(exc).code]
    finally:
        if client:client.close()
        result.update(generation_calls=client.generation_attempts if client else 0,wall_seconds=time.monotonic()-started)
        result.pop('network_calls',None)
        if acquired and 'GEMINI_TIMEOUT_NO_RETRY' not in result['errors']:lock.unlink()
        (folder/'result.json').write_text(json.dumps(result,ensure_ascii=True,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True));return 1 if result['errors'] else 0
if __name__=='__main__':raise SystemExit(main())
