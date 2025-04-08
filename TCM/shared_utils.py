import os
import joblib
import streamlit as st


def ensure_model_loaded():
    model_path = "models/tcm_diagnosis_model.pkl"
    if "tcm_model" not in st.session_state or st.session_state.tcm_model is None:
        if os.path.exists(model_path):
            st.session_state.tcm_model = joblib.load(model_path)
        else:
            st.session_state.tcm_model = None

