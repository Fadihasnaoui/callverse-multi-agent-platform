from pathlib import Path
import http.client,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quality_analyst.release_config import settings
api_port=os.getenv('QUALITY_ANALYST_API_PORT')
c=http.client.HTTPConnection('127.0.0.1',int(api_port) if api_port else settings().streamlit_port,timeout=3)
try:
    c.request('GET','/health' if api_port else '/_stcore/health');r=c.getresponse()
    raise SystemExit(0 if r.status==200 and (api_port or r.read(100)==b'ok') else 1)
finally:c.close()
