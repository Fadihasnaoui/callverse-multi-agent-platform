"""Metadata/preflight only. Never pulls a model or starts generation."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quality_analyst.release_hardware import preflight
from quality_analyst.release_config import settings
from quality_analyst.ollama import LocalOllama,INSTRUCT_PROFILE,INSTRUCT_DIGEST,AdapterError
snapshot,errors,model=preflight();identity=None;s=settings()
# /api/show is metadata only and remains useful when RAM admission refuses.
if model and model.get('digest')==INSTRUCT_DIGEST:
    try:identity=LocalOllama(profile=INSTRUCT_PROFILE).inspect_model()
    except AdapterError as exc:errors.append(exc.code)
print(json.dumps({'snapshot':snapshot,'errors':errors,'identity':identity,'generation_calls':0,
    'cpu_configuration':{'preflight_gib':s.cpu_ram_preflight_gib,'reserve_gib':s.cpu_ram_reserve_gib,
        'num_ctx':s.ollama_num_ctx,'num_predict':s.ollama_num_predict,'timeout_seconds':s.generation_timeout}},indent=2))
raise SystemExit(1 if errors else 0)
