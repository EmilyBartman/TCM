# tcm_app.py (Main Streamlit App)
# -*- coding: utf-8 -*-

import streamlit as st
import uuid
from datetime import datetime
from utils.firebase_utils import init_firebase, upload_image_to_firebase, save_user_submission
from utils.translation import LANGUAGES, translate, set_language_selector
from utils.gpt_diagnosis import run_gpt_diagnosis

# Set page config and language setup
st.set_page_config(page_title="TCM Health App", layout="wide")
db, bucket = init_firebase()
target_lang = set_language_selector()

# Page navigation
pages = [
    "Educational Content",
    "Tongue Health Check",
    "Medical Review Dashboard",
    "Submission History",
    "About & Disclaimer"
]
page = st.sidebar.radio("Navigate", pages)



# ------------------------------
# EDUCATIONAL CONTENT 
# ------------------------------
if page == "Educational Content":
    target_lang = languages[st.session_state.selected_lang]
    st.title(translate("🌿 Traditional Chinese Medicine (TCM) Education", target_lang))

    st.header(translate("Foundations of TCM", target_lang))
    st.markdown(translate("""
- **Yin & Yang**: Balance of opposing but complementary forces.
- **Qi (Chi)**: Vital life energy.
- **Five Elements**: Wood, Fire, Earth, Metal, Water—linked to organs/emotions.
- **Diagnostic Tools**: Pulse, tongue, face, symptom observation.
- **Modalities**: Acupuncture, herbal therapy, dietary therapy, Qi Gong.
""", target_lang))

    st.header(translate("🔎 Why the Tongue Matters in TCM", target_lang))
    st.markdown(translate("""
In Traditional Chinese Medicine, the tongue is seen as a mirror to the body’s internal state. Its color, shape, moisture, coating, and movement all provide clues about organ function and systemic imbalances.

**What the tongue reveals:**
- **Tongue Body Color**: Reflects blood, Qi, and organ health
- **Tongue Shape**: Can indicate excess or deficiency syndromes
- **Tongue Coating**: Suggests digestive heat, cold, or dampness
- **Moisture Level**: Linked to fluid metabolism and Yin/Yang balance
- **Tongue Movement**: Trembling or deviation may signal wind or weakness

Western medicine may not commonly use tongue inspection diagnostically, but it does correlate with:
- Anemia (pale tongue)
- Dehydration (dry tongue)
- Oral candidiasis (thick white coating)
- Circulatory issues (bluish-purple tongue)
""", target_lang))

    st.header(translate("🌐 Bridging TCM and Western Medicine", target_lang))
    st.markdown(translate("""
Traditional Chinese Medicine (TCM) and Western medicine differ in philosophy and methods but can be complementary:

| Concept | TCM Interpretation | Western Medicine Analogy |
|--------|---------------------|---------------------------|
| Qi (Vital Energy) | Flow of life energy through meridians | Nervous & Circulatory System activity |
| Yin/Yang | Balance of cold/hot, passive/active forces | Homeostasis (e.g., hormonal balance) |
| Tongue Diagnosis | Reflects internal organ status | Inflammation markers, dehydration, anemia |
| Syndrome Differentiation | Pattern-based holistic assessment | Evidence-based diagnosis (labs, scans) |

Integrative medicine combines both paradigms to enhance wellness, prevention, and personalized care.
""", target_lang))

    st.header(translate("TCM Syndrome Library", target_lang))
    with st.expander(translate("🔎 Click to view 8 Major Tongue Syndromes and Signs", target_lang)):
        st.markdown(translate("""
**Qi Deficiency**: Fatigue, pale tongue, short breath  
**Damp Retention**: Bloating, sticky tongue coat  
**Blood Stasis**: Sharp pain, purple tongue  
**Qi Stagnation**: Emotional blockage, rib pain  
**Damp Heat**: Yellow tongue coat, foul smell  
**Yang Deficiency**: Cold limbs, low energy  
**Yin Deficiency**: Dry mouth, night sweats  
**Blood Deficiency**: Pale lips, dizziness
""", target_lang))

    with st.expander(translate("📚 Recommended Reading", target_lang)):
        st.markdown(translate("""
- *Foundations of Chinese Medicine* - Giovanni Maciocia  
- *Healing with Whole Foods* - Paul Pitchford  
- *The Web That Has No Weaver* - Ted J. Kaptchuk  
- [WHO on TCM](https://www.who.int/health-topics/traditional-complementary-and-integrative-medicine)  
- [PubMed on TCM Research](https://pubmed.ncbi.nlm.nih.gov/?term=traditional+chinese+medicine)
""", target_lang))



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

        st.info("🧠 Sending image and data to GPT-4o for TCM diagnosis...")
        gpt_response = run_gpt_diagnosis(user_inputs, temp_path)

        if gpt_response:
            st.subheader("🤖 GPT-4o Diagnosis Result")
            st.code(gpt_response, language="json")


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

# ------------------------------
# SUBMISSION HISTORY
# ------------------------------
elif page == "Submission History":
    st.title("📜 My Tongue Scan History")
    if st.session_state.submissions:
        df = pd.DataFrame(st.session_state.submissions)
        st.dataframe(df)

        if "is_correct" in df.columns:
            correct_count = df["is_correct"].sum()
            total_feedback = df["is_correct"].notna().sum()
            if total_feedback > 0:
                accuracy = round((correct_count / total_feedback) * 100, 2)
                st.metric("📊 Model Accuracy (based on feedback)", f"{accuracy}%")

            st.subheader("📈 Accuracy Over Time")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            daily = df[df["is_correct"].notna()].groupby(df["timestamp"].dt.date)["is_correct"].mean()
            st.line_chart(daily)

            st.subheader("🧪 Accuracy by TCM Syndrome")
            if "prediction_TCM" in df.columns:
                by_syndrome = df[df["is_correct"].notna()].groupby("prediction_TCM")["is_correct"].mean().sort_values(ascending=False)
                st.bar_chart(by_syndrome)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "my_tongue_scans.csv", "text/csv")
    else:
        st.info("You haven't submitted any scans yet.")

# ------------------------------
#  ABOUT & DISCLAIMER 
# ------------------------------
elif page == "About & Disclaimer":
    st.title(translate("About This App", target_lang))
    about_text = '''
    This app is built for:
    - Educating users about TCM tongue diagnostics
    - Demonstrating how AI may assist in early wellness screening
    - Researching global health variations using tongue + symptom data

    Data Usage: All uploaded data is securely stored and used anonymously for improving model prediction.

    Disclaimer: This tool is for educational purposes only. It does not replace medical diagnosis or professional care.
    '''
    st.markdown(translate(about_text, target_lang))
