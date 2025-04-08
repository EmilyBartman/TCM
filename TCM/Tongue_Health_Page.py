# Inside render_tongue_health_check()
from app_tcm import ensure_model_loaded  # if your main file is named app_tcm.py
ensure_model_loaded()  # 🔄 Force reload model from file into session state

# Optional debug info
st.caption(f"🔥 [Debug] Model state: {'loaded' if 'tcm_model' in st.session_state and st.session_state.tcm_model else 'missing'}")
st.caption(f"📁 Model file present: {os.path.exists('models/tcm_diagnosis_model.pkl')}")
st.caption(f"🧬 Feature length: {len(features)}")
