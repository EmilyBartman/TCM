import os
import joblib
import streamlit as st

def ensure_model_loaded():
    if "tcm_model" not in st.session_state or st.session_state.tcm_model is None:
        model_path = "models/tcm_diagnosis_model.pkl"
        if os.path.exists(model_path):
            st.session_state.tcm_model = joblib.load(model_path)
        else:
            from app_tcm import retrain_model_from_firestore, get_firestore_client
            db = get_firestore_client()
            retrain_model_from_firestore(db)
            if os.path.exists(model_path):
                st.session_state.tcm_model = joblib.load(model_path)
