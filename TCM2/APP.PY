# tcm_app.py (Main Streamlit App)
# -*- coding: utf-8 -*-

import streamlit as st
import uuid
from datetime import datetime
from utils.firebase_utils import init_firebase, upload_image_to_firebase, save_user_submission
from utils.model_utils import extract_features, load_model, predict_with_model, compare_user_vs_model, get_remedies
from utils.translation import LANGUAGES, translate, set_language_selector

# Set page config and language setup
st.set_page_config(page_title="TCM Health App", layout="wide")
db, bucket = init_firebase()
target_lang = set_language_selector()

# Page navigation
pages = [
    "Tongue Health Check",
    "Medical Review Dashboard"
]
page = st.sidebar.radio("Navigate", pages)

# ------------------------------
# Tongue Health Check
# ------------------------------
if page == "Tongue Health Check":
    st.title(translate("👅 Tongue Diagnosis Tool", target_lang))

    with st.form("tongue_upload_form"):
        uploaded_img = st.file_uploader("Upload Tongue Image", type=["jpg", "jpeg", "png"])
        symptoms = st.multiselect("Select Symptoms", ["Fatigue", "Stress", "Stomach ache", "Dizziness"])

        tongue_color = st.selectbox("Tongue Color", ["Red", "Pale", "Purple"])
        tongue_shape = st.selectbox("Tongue Shape", ["Swollen", "Thin", "Tooth-marked"])
        tongue_coating = st.selectbox("Tongue Coating", ["White", "Yellow", "None"])
        tongue_moisture = st.selectbox("Moisture Level", ["Dry", "Wet", "Normal"])
        tongue_bumps = st.selectbox("Tongue Bumps", ["Smooth", "Raised", "Prominent"])

        heart_rate = st.number_input("Heart Rate (bpm)", 40, 180)
        sleep_hours = st.slider("Hours of Sleep", 0, 12, 6)
        stress_level = st.slider("Stress Level", 0, 10, 5)
        hydration = st.radio("Do you feel thirsty often?", ["Yes", "No"])
        bowel = st.selectbox("Bowel Regularity", ["Regular", "Irregular"])
        medication = st.text_input("Current Medication (optional)")

        consent = st.checkbox("I consent to the use of my data for research.")
        submit = st.form_submit_button("Analyze")

    if submit:
        if not uploaded_img or not consent:
            st.warning("Please upload image and give consent.")
            st.stop()

        submission_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Save image
        url, temp_path = upload_image_to_firebase(uploaded_img, submission_id, bucket)

        # Save user input
        user_inputs = {
            "symptoms": symptoms,
            "tongue_characteristics": {
                "color": tongue_color,
                "shape": tongue_shape,
                "coating": tongue_coating,
                "moisture": tongue_moisture,
                "bumps": tongue_bumps
            },
            "vitals": {
                "heart_rate": heart_rate,
                "sleep_hours": sleep_hours,
                "hydration": hydration,
                "stress_level": stress_level,
                "bowel_movement": bowel,
                "medication": medication
            },
            "consent_given": consent
        }
        save_user_submission(submission_id, timestamp, url, user_inputs, db)

        # Run model
        features = extract_features(temp_path)
        model = load_model()
        prediction, confidence = predict_with_model(model, features)

        model_outputs = {
            "symptoms": ["Fatigue"],
            "tongue_characteristics": {
                "color": "pale",
                "shape": "swollen",
                "coating": "white",
                "moisture": "dry",
                "bumps": "prominent"
            },
            "tcm_syndrome": prediction,
            "western_equiv": "Anemia",
            "remedies": get_remedies(prediction)
        }

        diff, score = compare_user_vs_model(user_inputs, model_outputs)

        db.collection("model_outputs").document(submission_id).set({
            "submission_id": submission_id,
            "model_outputs": model_outputs,
            "confidence_score": score,
            "features": features,
            "timestamp": timestamp
        })
        db.collection("submission_diffs").document(submission_id).set({
            "submission_id": submission_id,
            "differences": diff,
            "confidence_score": score,
            "timestamp": timestamp
        })

        st.success(f"Prediction: {prediction} | Confidence: {score}%")
        st.json(model_outputs)
        st.json(diff)

# ------------------------------
# Medical Review Dashboard
# ------------------------------
elif page == "Medical Review Dashboard":
    st.title("🧠 Medical Review Dashboard")
    st.info("Select a submission to review and give expert feedback.")

    docs = db.collection("submission_diffs").stream()
    ids = [d.id for d in docs]
    selected_id = st.selectbox("Submission ID", ids)

    if selected_id:
        user_doc = db.collection("tongue_submissions").document(selected_id).get().to_dict()
        model_doc = db.collection("model_outputs").document(selected_id).get().to_dict()
        diff_doc = db.collection("submission_diffs").document(selected_id).get().to_dict()

        st.subheader("📄 User Input")
        st.json(user_doc["user_inputs"])

        st.subheader("🤖 Model Output")
        st.json(model_doc["model_outputs"])

        st.subheader("⚖️ Differences")
        st.json(diff_doc["differences"])

        st.subheader("🧬 Feedback")
        agree = st.radio("Do you agree with model?", ["Yes", "Partially", "No"])
        notes = st.text_area("Correction notes")
        if st.button("Submit Feedback"):
            db.collection("medical_feedback").document(selected_id).set({
                "submission_id": selected_id,
                "agreement": agree,
                "correction_notes": notes,
                "timestamp": datetime.utcnow().isoformat()
            })
            st.success("Feedback saved.")

        with st.expander("🔄 Retrain From Feedback"):
            from utils.retrain import retrain_model_from_feedback
            if st.button("🔄 Retrain Now"):
                retrain_model_from_feedback(db)
