# tcm_app.py (Main Streamlit App)
# -*- coding: utf-8 -*-

import streamlit as st
import uuid
import openai
from datetime import datetime
from utils.firebase_utils import init_firebase, upload_image_to_firebase, save_user_submission
from utils.translation import LANGUAGES, translate, set_language_selector
from utils.gpt_diagnosis import run_gpt_diagnosis
from torchvision import models, transforms
from PIL import Image
import torch
# GPT-4o key
openai.api_key = st.secrets["openai"]["api_key"]

# Firebase config
firebase_config = dict(st.secrets["firebase"])

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
    target_lang = LANGUAGES[st.session_state.selected_lang]
    st.title(translate("🌿 Traditional Chinese Medicine (TCM) Education", target_lang))

    st.markdown("### 👥 For General Users & Beginners")
    st.markdown(translate("""
**What Is TCM?**
Traditional Chinese Medicine (TCM) is a holistic approach to health practiced for over 2,500 years. It focuses on restoring balance in the body rather than just treating symptoms.

**Core Principles:**
- **Yin & Yang**: Balance of opposite energies (cold/hot, passive/active)
- **Qi (Chi)**: Vital energy flowing through the body
- **Five Elements**: Wood, Fire, Earth, Metal, Water — each linked to body systems and emotions
- **How Diagnosis Works**: Practitioners look at your pulse, tongue, skin, and ask about your lifestyle
- **Common Treatments**: Herbal remedies, acupuncture, gentle movement (Tai Chi/Qi Gong), and dietary therapy
""", target_lang))

    st.markdown("### 👅 Tongue Diagnosis Basics")
    st.markdown(translate("""
In TCM, your tongue offers a visual map of your internal health. Changes in its appearance can signal imbalances before other symptoms appear.

**What to look for:**
- **Color**: Pale (deficiency), Red (heat), Purple (stagnation)
- **Shape**: Swollen (damp retention), Thin (Yin deficiency)
- **Coating**: White (cold), Yellow (heat), Thick (digestive imbalance)
- **Moisture**: Dry (Yin deficiency), Wet (Yang deficiency)
- **Movement**: Trembling or stiff may indicate internal disturbances

These insights help tailor your care plan.
""", target_lang))

    st.markdown("---")
    st.markdown("### 🩺 For Medical Professionals & Practitioners")
    st.markdown(translate("""
**Diagnostic Framework in TCM:**
Tongue inspection is one of the four key diagnostic methods in TCM, alongside pulse taking, observation, and inquiry. It offers insights into the state of internal organs and their pathological changes.

**Clinical Tongue Indicators:**
- **Color**: Indicates blood and Qi flow, thermal state
- **Shape & Texture**: Points to fluid metabolism, organ deficiency/excess
- **Coating**: Derived from stomach Qi; helps assess dampness, heat, cold
- **Sub-lingual veins**: Often inspected for blood stasis patterns

**Crosswalk to Western Medicine:**
| TCM Observation       | Possible Western Interpretation         |
|----------------------|------------------------------------------|
| Pale tongue           | Anemia, blood deficiency                |
| Red with yellow coat  | Inflammatory conditions, infection      |
| Purple tongue         | Circulatory issues                      |
| Thick white coating   | Oral candidiasis, poor digestion        |

**Applications in Integrative Care:**
Tongue signs may support early detection of systemic imbalances before lab markers change. Use cases include GI disorders, hormonal imbalance, and fatigue-related syndromes.
""", target_lang))

    st.markdown("### 🧠 AI & Tongue Diagnostics")
    st.markdown(translate("""
This app uses AI (specifically GPT-4o) to analyze tongue images and user-reported symptoms. While not a replacement for clinical judgment, it offers a research-based wellness screening tool that may help with:
- Tracking progress over time
- Generating wellness recommendations
- Supporting training and education in visual diagnosis
""", target_lang))

    st.header(translate("TCM Syndrome Reference", target_lang))
    with st.expander(translate("📘 View 8 Major Syndromes and Their Signs", target_lang)):
        st.markdown(translate("""
**Qi Deficiency**: Fatigue, pale tongue, low voice  
**Damp Retention**: Bloating, sticky tongue coat  
**Blood Stasis**: Sharp pain, purple tongue  
**Qi Stagnation**: Mood swings, rib-side pain  
**Damp Heat**: Yellow greasy coating, foul odor  
**Yang Deficiency**: Cold limbs, weak pulse  
**Yin Deficiency**: Night sweats, dry mouth  
**Blood Deficiency**: Pale lips, dizziness
""", target_lang))

    with st.expander(translate("📚 Recommended Reading", target_lang)):
        st.markdown(translate("""
- *Foundations of Chinese Medicine* – Giovanni Maciocia  
- *Healing with Whole Foods* – Paul Pitchford  
- *The Web That Has No Weaver* – Ted Kaptchuk  
- [WHO on TCM](https://www.who.int/health-topics/traditional-complementary-and-integrative-medicine)  
- [PubMed TCM Research](https://pubmed.ncbi.nlm.nih.gov/?term=traditional+chinese+medicine)
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
        
            try:
                db.collection("gpt_diagnoses").document(submission_id).set({
                    "submission_id": submission_id,
                    "timestamp": timestamp,
                    "image_url": url,
                    "user_inputs": user_inputs,
                    "gpt_response": gpt_response
                })
                st.success("📝 GPT diagnosis result saved to Firestore.")
            except Exception as e:
                st.warning("⚠️ Failed to save GPT diagnosis result.")
                st.exception(e)


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
        gpt_doc = db.collection("gpt_diagnoses").document(selected_id).get().to_dict()
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
    if gpt_doc:
        st.subheader("🧠 GPT-4o Diagnosis Result")
    
        gpt_data = gpt_doc.get("gpt_response", "")
        st.code(gpt_data, language="json")
    else:
        st.info("GPT-4o response not found for this submission.")
    
    with st.expander("🔄 Retrain From Feedback"):
        from utils.retrain import retrain_model_from_feedback
        if st.button("🔄 Retrain Now"):
            retrain_model_from_feedback(db)

# ------------------------------
# SUBMISSION HISTORY
# ------------------------------
elif page == "Submission History":
    st.title("📊 Model & App Performance Dashboard")

    try:
        # Fetch Firestore data
        usage_docs = db.collection("tongue_submissions").stream()
        gpt_docs = db.collection("gpt_diagnoses").stream()
        feedback_docs = db.collection("medical_feedback").stream()

        usage_data = [doc.to_dict() for doc in usage_docs]
        gpt_data = [doc.to_dict() for doc in gpt_docs]
        feedback_data = [doc.to_dict() for doc in feedback_docs]

        # Submissions Over Time
        st.subheader("📈 Tongue Submissions Over Time")
        df_usage = pd.DataFrame(usage_data)
        df_usage["timestamp"] = pd.to_datetime(df_usage["timestamp"])
        last_30 = datetime.utcnow() - pd.Timedelta(days=30)
        df_usage = df_usage[df_usage["timestamp"] >= last_30]

        if not df_usage.empty:
            df_usage["timestamp"] = pd.to_datetime(df_usage["timestamp"])
            count_per_day = df_usage.groupby(df_usage["timestamp"].dt.date).size()
            st.line_chart(count_per_day.rename("Submissions"))

        # Feedback Summary
        st.subheader("🧬 Expert Feedback Trends")
        df_fb = pd.DataFrame(feedback_data)
        df_fb["timestamp"] = pd.to_datetime(df_fb["timestamp"])
        df_fb = df_fb[df_fb["timestamp"] >= last_30]


        if not df_fb.empty and "agreement" in df_fb.columns:
            agree_dist = df_fb["agreement"].value_counts()
            st.bar_chart(agree_dist.rename("Agreement Distribution"))

            df_fb["timestamp"] = pd.to_datetime(df_fb["timestamp"])
            trend = df_fb.groupby(df_fb["timestamp"].dt.date)["agreement"].apply(lambda x: (x == "Yes").mean())
            st.line_chart(trend.rename("% Agreement (Yes)"))

        # GPT Diagnoses Summary
        st.subheader("🧠 Recent GPT-4o Diagnoses")
        df_gpt = pd.DataFrame(gpt_data)
        df_gpt["timestamp"] = pd.to_datetime(df_gpt["timestamp"])
        df_gpt = df_gpt[df_gpt["timestamp"] >= last_30]
        if not df_gpt.empty:
            df_gpt["timestamp"] = pd.to_datetime(df_gpt["timestamp"])
            recent = df_gpt.sort_values("timestamp", ascending=False).head(10)
            st.dataframe(recent[["timestamp", "submission_id", "gpt_response"]])

            csv = df_gpt.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download All GPT Diagnoses", csv, "gpt_diagnoses.csv", "text/csv")

        if df_usage.empty and df_gpt.empty and df_fb.empty:
            st.info("No diagnostic data has been submitted yet.")

    except Exception as e:
        st.error("⚠️ Failed to load metrics from Firestore.")
        st.exception(e)


# ------------------------------
#  ABOUT & DISCLAIMER 
# ------------------------------
elif page == "About & Disclaimer":
    st.title(translate("About This App", target_lang))
    about_text = '''
        **Wise Tongue: AI-Powered TCM Health Companion**
        
        This application is designed to:
        - Provide accessible education on Traditional Chinese Medicine (TCM), with a focus on tongue-based diagnostics
        - Allow users to upload tongue images and symptoms for AI-assisted analysis
        - Use multimodal models (e.g., GPT-4o) to suggest potential TCM patterns, Western analogies, and wellness recommendations
        - Enable healthcare professionals to review, validate, and improve AI predictions over time
        
        **Data Usage & Privacy**
        - All user data, including images and self-reported symptoms, is securely stored in Firebase
        - Data is used solely for improving the model's diagnostic accuracy and understanding global health trends
        - Personal identities are not collected, and all data is reviewed anonymously
        - Professional feedback is used to retrain the model and enhance future predictions
        
        **Disclaimer**
        This application does not provide medical diagnosis or treatment. It is intended for educational, research, and wellness support purposes only. Users should consult qualified healthcare providers for any medical concerns.
        '''
    st.markdown(translate(about_text, target_lang))
