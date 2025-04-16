# tcm_app.py (Main Streamlit App)
# -*- coding: utf-8 -*-

import streamlit as st
import uuid
import openai
import json
import io
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
# Page navigation
pages = [
    "Educational Content",
    "Tongue Health Check",
    "Medical Review Dashboard",
    "Submission History",
    "About & Disclaimer"
]

# Create mapping of internal keys -> translated display names
page_options = {p: translate(p, target_lang) for p in pages}

# Initialize session state
if "selected_page" not in st.session_state:
    st.session_state.selected_page = pages[0]

# Translated names for UI
display_names = list(page_options.values())

# Sidebar radio with translated labels
selected_display = st.sidebar.radio(
    translate("Navigate", target_lang),
    display_names,
    index=display_names.index(page_options[st.session_state.selected_page]),
    key="page_navigation"
)

# Set selected page from translated label
for internal, display in page_options.items():
    if display == selected_display:
        st.session_state.selected_page = internal
        break

# Final selected page
page = st.session_state.selected_page



# ------------------------------
# EDUCATIONAL CONTENT 
# ------------------------------
if page == "Educational Content":
    target_lang = LANGUAGES[st.session_state.selected_lang]
    st.title(translate("🌿 Traditional Chinese Medicine (TCM) Education", target_lang))

    st.markdown(f"### 👥 **{translate('For General Users & Beginners', target_lang)}**")
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

    st.markdown(f"### 👅 **{translate('Tongue Diagnosis Basics', target_lang)}**")

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
    st.markdown(f"### 🩺 **{translate('For Medical Professionals & Practitioners', target_lang)}**")
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

    st.markdown(f"### 🧠 **{translate('AI & Tongue Diagnostics', target_lang)}**")
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

    # Upload and preview image BEFORE the form
    st.markdown(translate("Upload Tongue Image", target_lang))
    st.markdown(translate("Drag and drop a file below. Limit 200MB per file • JPG, JPEG, PNG", target_lang))
    
    # Upload and preview image BEFORE the form
    st.markdown(translate("Upload Tongue Image", target_lang))
    st.markdown(translate("Drag and drop a file below. Limit 200MB per file • JPG, JPEG, PNG", target_lang))
    
    # Only call file_uploader ONCE and manage session
    if "uploaded_img" not in st.session_state:
        st.session_state.uploaded_img = None
    
    uploaded_img = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_img is not None:
        st.session_state.uploaded_img = uploaded_img
    
    uploaded_img = st.session_state.uploaded_img


    
    # Preview image if available
    if uploaded_img:
        try:
            uploaded_img.seek(0)
            image_bytes = uploaded_img.read()
            img = Image.open(io.BytesIO(image_bytes))
            st.image(img, caption=translate("Preview of Uploaded Tongue Image", target_lang), width=300)
        except Exception as e:
            st.warning(translate("⚠️ Unable to preview uploaded image.", target_lang))
            st.exception(e)


    with st.form("tongue_upload_form"):
                
        symptoms = st.multiselect(
            translate("Select Symptoms", target_lang),
            [translate(opt, target_lang) for opt in ["Fatigue", "Stress", "Stomach ache", "Dizziness"]]
        )

        tongue_color = st.selectbox(
            translate("Tongue Color", target_lang),
            [translate(opt, target_lang) for opt in ["Red", "Pale", "Purple"]]
        )
        tongue_shape = st.selectbox(
            translate("Tongue Shape", target_lang),
            [translate(opt, target_lang) for opt in ["Swollen", "Thin", "Tooth-marked"]]
        )
        tongue_coating = st.selectbox(
            translate("Tongue Coating", target_lang),
            [translate(opt, target_lang) for opt in ["White", "Yellow", "None"]]
        )
        tongue_moisture = st.selectbox(
            translate("Moisture Level", target_lang),
            [translate(opt, target_lang) for opt in ["Dry", "Wet", "Normal"]]
        )
        tongue_bumps = st.selectbox(
            translate("Tongue Bumps", target_lang),
            [translate(opt, target_lang) for opt in ["Smooth", "Raised", "Prominent"]]
        )

        heart_rate = st.number_input(translate("Heart Rate (bpm)", target_lang), 40, 180)
        sleep_hours = st.slider(translate("Hours of Sleep", target_lang), 0, 12, 6)
        stress_level = st.slider(translate("Stress Level", target_lang), 0, 10, 5)
        hydration = st.radio(
            translate("Do you feel thirsty often?", target_lang),
            [translate(opt, target_lang) for opt in ["Yes", "No"]]
        )
        bowel = st.selectbox(
            translate("Bowel Regularity", target_lang),
            [translate(opt, target_lang) for opt in ["Regular", "Irregular"]]
        )
        medication = st.text_input(translate("Current Medication (optional)", target_lang))

        consent = st.checkbox(translate("I consent to the use of my data for research.", target_lang))
        submit = st.form_submit_button(translate("Analyze", target_lang))
        if submit:
            st.session_state.form_submitted = True
            uploaded_img = st.session_state.get("uploaded_img", None)
 


    if submit:
        # Check for image and consent before continuing
        if not uploaded_img or not consent:
            st.warning(translate("Please upload image and give consent.", target_lang))
            st.stop()
        
        st.session_state.form_submitted = True  # Optional: to block language switch rerun


        submission_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        url, temp_path = upload_image_to_firebase(uploaded_img, submission_id, bucket)

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

        st.info(translate("🧠 Processing data with GPT-4o for TCM prediction...", target_lang))
        gpt_response = run_gpt_diagnosis(user_inputs, temp_path)
        if gpt_response:
            if isinstance(gpt_response, dict):
                st.subheader(translate("Prediction Result", target_lang))
                st.markdown(f"**🩺 TCM Syndrome:** {gpt_response.get('tcm_syndrome', 'N/A')}")
                st.markdown(f"**💊 Western Equivalent:** {gpt_response.get('western_equivalent', 'N/A')}")
                st.markdown("**🌿 Remedies:**")
                for r in gpt_response.get("remedies", []):
                    st.markdown(f"- {r}")
                st.markdown(f"**⚖️ Discrepancies:** {gpt_response.get('discrepancies', 'N/A')}")
                st.markdown(f"**📊 Confidence Score:** {gpt_response.get('confidence', 'N/A')}%")
            else:
                st.subheader(translate("Prediction Result", target_lang))
                st.warning("Could not parse structured response. Displaying raw output:")
                st.write(gpt_response)


            try:
                db.collection("gpt_diagnoses").document(submission_id).set({
                    "submission_id": submission_id,
                    "timestamp": timestamp,
                    "image_url": url,
                    "user_inputs": user_inputs,
                    "gpt_response": gpt_response
                })
                st.success(translate("📝 Prediction result saved to Firestore.", target_lang))
            except Exception as e:
                st.warning(translate("⚠️ Failed to save prediction result.", target_lang))
                st.exception(e)
            
            st.markdown("---")
            st.markdown(translate(
                f"📝 **Disclaimer:**\n\n"
                f"The insights above are based primarily on the symptoms and tongue characteristics you reported — "
                f"the AI model is not trained to directly interpret tongue images. However, your submitted image and input data "
                f"are securely stored and may help improve future versions of this tool.\n\n"
                f"This is not a medical diagnosis. For any health concerns, please consult a licensed healthcare provider.",
            target_lang))

    # Reset just_submitted flag at the end of the run
    st.session_state.form_submitted = False




# ------------------------------
# Medical Review Dashboard
# ------------------------------
elif page == "Medical Review Dashboard":
    st.title(translate("🧠 Medical Review Dashboard", target_lang))
    st.info(translate("Select a submission to review and give expert feedback.", target_lang))

    docs = db.collection("submission_diffs").stream()
    ids = [d.id for d in docs]
    selected_id = st.selectbox(translate("Submission ID", target_lang), ids)

    if selected_id:
        user_doc = db.collection("tongue_submissions").document(selected_id).get().to_dict()
        gpt_doc = db.collection("gpt_diagnoses").document(selected_id).get().to_dict()
        model_doc = db.collection("model_outputs").document(selected_id).get().to_dict()
        diff_doc = db.collection("submission_diffs").document(selected_id).get().to_dict()

        st.subheader(translate("📄 User Input", target_lang))
        st.json(user_doc["user_inputs"])

        st.subheader(translate("🤖 Model Output", target_lang))
        st.json(model_doc["model_outputs"])

        st.subheader(translate("⚖️ Differences", target_lang))
        st.json(diff_doc["differences"])

        st.subheader(translate("🧬 Feedback", target_lang))
        agree = st.radio(
            translate("Do you agree with model?", target_lang),
            [translate(opt, target_lang) for opt in ["Yes", "Partially", "No"]]
        )
        notes = st.text_area(translate("Correction notes", target_lang))
        if st.button(translate("Submit Feedback", target_lang)):
            db.collection("medical_feedback").document(selected_id).set({
                "submission_id": selected_id,
                "agreement": agree,
                "correction_notes": notes,
                "timestamp": datetime.utcnow().isoformat()
            })
            st.success(translate("Feedback saved.", target_lang))

    if gpt_doc:
        st.subheader(translate("🧠 GPT-4o Diagnosis Result", target_lang))
        gpt_data = gpt_doc.get("gpt_response", "")
        if gpt_doc:
            st.subheader(translate("🧠 GPT-4o Diagnosis Result", target_lang))
            gpt_data = gpt_doc.get("gpt_response", "")
            
            if isinstance(gpt_data, dict):
                st.markdown(f"**🩺 TCM Syndrome:** {gpt_data.get('tcm_syndrome', 'N/A')}")
                st.markdown(f"**💊 Western Equivalent:** {gpt_data.get('western_equivalent', 'N/A')}")
                st.markdown("**🌿 Remedies:**")
                for r in gpt_data.get("remedies", []):
                    st.markdown(f"- {r}")
                st.markdown(f"**⚖️ Discrepancies:** {gpt_data.get('discrepancies', 'N/A')}")
                st.markdown(f"**📊 Confidence Score:** `{gpt_data.get('confidence', 'N/A')}%`")
            else:
                st.warning(translate("Could not parse structured GPT response. Displaying raw text:", target_lang))
                st.write(gpt_data)
        
            raw_gpt = gpt_doc.get("gpt_response", "")
            try:
                gpt_data = raw_gpt if isinstance(raw_gpt, dict) else json.loads(raw_gpt)
            except Exception:
                gpt_data = raw_gpt 

    else:
        st.info(translate("GPT-4o response not found for this submission.", target_lang))


    with st.expander(translate("🔄 Retrain From Feedback", target_lang)):
        from utils.retrain import retrain_model_from_feedback
        if st.button(translate("🔄 Retrain Now", target_lang)):
            retrain_model_from_feedback(db)


# ------------------------------
# SUBMISSION HISTORY
# ------------------------------
elif page == "Submission History":
    st.title(translate("📊 Model & App Performance Dashboard", target_lang))

    try:
        usage_docs = db.collection("tongue_submissions").stream()
        gpt_docs = db.collection("gpt_diagnoses").stream()
        feedback_docs = db.collection("medical_feedback").stream()

        usage_data = [doc.to_dict() for doc in usage_docs]
        gpt_data = [doc.to_dict() for doc in gpt_docs]
        feedback_data = [doc.to_dict() for doc in feedback_docs]

        st.subheader(translate("📈 Tongue Submissions Over Time", target_lang))
        df_usage = pd.DataFrame(usage_data)
        df_usage["timestamp"] = pd.to_datetime(df_usage["timestamp"])
        last_30 = datetime.utcnow() - pd.Timedelta(days=30)
        df_usage = df_usage[df_usage["timestamp"] >= last_30]

        if not df_usage.empty:
            count_per_day = df_usage.groupby(df_usage["timestamp"].dt.date).size()
            st.line_chart(count_per_day.rename(translate("Submissions", target_lang)))

        st.subheader(translate("🧬 Expert Feedback Trends", target_lang))
        df_fb = pd.DataFrame(feedback_data)
        df_fb["timestamp"] = pd.to_datetime(df_fb["timestamp"])
        df_fb = df_fb[df_fb["timestamp"] >= last_30]

        if not df_fb.empty and "agreement" in df_fb.columns:
            agree_dist = df_fb["agreement"].value_counts()
            agree_dist.index = [translate(opt, target_lang) for opt in agree_dist.index]
            st.bar_chart(agree_dist.rename(translate("Agreement Distribution", target_lang)))

            trend = df_fb.groupby(df_fb["timestamp"].dt.date)["agreement"].apply(
                lambda x: (x == "Yes").mean()
            )
            st.line_chart(trend.rename(translate("% Agreement (Yes)", target_lang)))

        st.subheader(translate("🧠 Recent GPT-4o Diagnoses", target_lang))
        df_gpt = pd.DataFrame(gpt_data)
        df_gpt["timestamp"] = pd.to_datetime(df_gpt["timestamp"])
        df_gpt = df_gpt[df_gpt["timestamp"] >= last_30]
        if not df_gpt.empty:
            recent = df_gpt.sort_values("timestamp", ascending=False).head(10)
            st.dataframe(recent[["timestamp", "submission_id", "gpt_response"]])

            csv = df_gpt.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=translate("⬇️ Download All GPT Diagnoses", target_lang),
                data=csv,
                file_name="gpt_diagnoses.csv",
                mime="text/csv"
            )

        if df_usage.empty and df_gpt.empty and df_fb.empty:
            st.info(translate("No diagnostic data has been submitted yet.", target_lang))

    except Exception as e:
        st.error(translate("⚠️ Failed to load metrics from Firestore.", target_lang))
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
