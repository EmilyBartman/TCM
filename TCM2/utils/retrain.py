# utils/retrain.py
import os
import joblib
import streamlit as st
from sklearn.linear_model import LogisticRegression

def retrain_model_from_feedback(db):
    st.info("🔁 Fetching corrections from medical feedback...")
    corrections = db.collection("medical_feedback").stream()
    training_data = []

    for doc in corrections:
        fb = doc.to_dict()
        if fb.get("agreement") == "Yes":
            continue
        sid = fb["submission_id"]

        features_doc = db.collection("model_outputs").document(sid).get()
        if features_doc.exists:
            features = features_doc.to_dict().get("features")
            label = fb.get("correction_notes", "").strip()
            if features and label:
                training_data.append((features, label))

    if not training_data:
        st.warning("❌ No disagreements with corrections to retrain.")
        return

    X = [x[0] for x in training_data]
    y = [x[1] for x in training_data]

    st.info(f"🔧 Retraining on {len(X)} labeled examples...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/tcm_diagnosis_model.pkl")
    st.success("✅ Retrained model saved to disk.")
