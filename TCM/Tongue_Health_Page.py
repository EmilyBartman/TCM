# Inside render_tongue_health_check()
import streamlit as st
from shared_utils import ensure_model_loaded

ensure_model_loaded()  # 🔄 Force reload model from file into session state

# Optional debug info
st.caption(f"🔥 [Debug] Model state: {'loaded' if 'tcm_model' in st.session_state and st.session_state.tcm_model else 'missing'}")
st.caption(f"📁 Model file present: {os.path.exists('models/tcm_diagnosis_model.pkl')}")
st.caption(f"🧬 Feature length: {len(features)}")
