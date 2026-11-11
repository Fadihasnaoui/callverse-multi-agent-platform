from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import streamlit as st
from quality_analyst.release_config import ConfigurationError,settings
try:
    settings()
    from quality_analyst.ozone_ui_v02 import main
    main()
except ConfigurationError as exc:
    st.error('Configuration invalide : '+str(exc))
    st.stop()
