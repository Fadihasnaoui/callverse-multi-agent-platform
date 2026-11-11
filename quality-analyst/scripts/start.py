"""Cross-platform entry point; validated .env feeds Streamlit and workers."""
from pathlib import Path
import os, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quality_analyst.release_config import settings,configure_logging,ConfigurationError

def main():
    try:
        s=settings();configure_logging();s.data_dir.mkdir(parents=True,exist_ok=True)
        # Import-time config and subprocess workers share the same ROOT/.env.
        os.environ['PYTHONDONTWRITEBYTECODE']='1'
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS']='false'
        os.environ['STREAMLIT_CLIENT_SHOW_ERROR_DETAILS']='none' if s.app_env=='production' else 'full'
        command=[sys.executable,'-B','-m','streamlit','run',str(ROOT/'scripts/app.py'),
            '--server.address','0.0.0.0','--server.port',str(s.streamlit_port),
            '--server.headless','true','--server.fileWatcherType','none',
            '--server.maxUploadSize','1','--theme.base','light',
            '--logger.level',s.log_level.lower()]
        os.execv(sys.executable,command)
    except ConfigurationError as exc:
        print('Configuration invalide : '+str(exc),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
