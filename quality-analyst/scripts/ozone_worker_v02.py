"""One v0.2 stage; no HTTP hotel client or optional UI dependency in .venv."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quality_analyst.release_config import settings
from quality_analyst.contracts import load_conversation_json,strict_json_loads
from quality_analyst.ozone_live.real_contracts import SearchResult
from quality_analyst.ai_providers import get_provider
from quality_analyst.interactive import write_json
from quality_analyst.ollama import compact,sha256

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',type=Path,required=True)
    folder=parser.parse_args().session.resolve()
    if not folder.is_relative_to((settings().data_dir/'ozone_v02_sessions').resolve()):raise ValueError('RESERVED_V02_SESSION_REQUIRED')
    data=strict_json_loads((folder/'input.json').read_bytes());reservation=strict_json_loads((folder/'reservation.json').read_bytes())
    if sha256(compact(data).encode())!=reservation['packet_sha256']:raise ValueError('INPUT_CHANGED')
    write_json(folder/'worker_started.json',{'maximum_generations':1,'automatic_retries':0})
    c=load_conversation_json(compact(data['conversation']))
    retrieval=SearchResult.model_validate_json(compact(data['retrieval'])) if data['retrieval'] else None
    provider=data.get('provider','ollama')
    expected=settings().gemini_model if provider=='gemini' else settings().ollama_model
    if data.get('model',expected)!=expected:raise ValueError('RESERVED_MODEL_CHANGED')
    get_provider(provider).execute(c,folder/'runtime',stage=data['stage'],context=data['context'],retrieval=retrieval,
        cloud_consent=data.get('cloud_consent',False),free_tier_confirmed=data.get('free_tier_confirmed',False))

if __name__=='__main__':main()
