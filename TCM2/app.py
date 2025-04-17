# tcm_app.py (Main Streamlit App)
# -*- coding: utf-8 -*-

import streamlit as st
import uuid
import openai
from utils.model_utils import get_remedies
import json
import io
import pandas as pd
from datetime import datetime
from utils.firebase_utils import init_firebase, upload_image_to_firebase, save_user_submission
from utils.translation import LANGUAGES, translate, set_language_selector
from utils.gpt_diagnosis import run_gpt_diagnosis
from utils.retrain import retrain_model_from_feedback
from torchvision import models, transforms
import requests
from io import BytesIO
from PIL import Image
import torch

# 🔑 API and Firebase setup
openai.api_key = st.secrets["openai"]["api_key"]
firebase_config = dict(st.secrets["firebase"])

# 🌐 App Configuration
st.set_page_config(page_title="TCM Health App", layout="wide")
# 🔧 Reduce top padding to fix whitespace
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

db, bucket = init_firebase()


# ✨ Improve Tabs Style
st.markdown("""
    <style>
    /* Make the tabs bigger and spaced */
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        padding: 12px 24px;
        border-bottom: 3px solid transparent;
    }
    /* Highlight active tab */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #ff4b4b; /* Bright red underline */
        color: #ff4b4b; /* Bright red text */
        font-weight: bold;
        background-color: #f9f9f9;
        border-radius: 8px 8px 0 0;
    }
    /* On hover over tabs */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# 🎨 Make selectbox (language) smaller and tighter
st.markdown("""
    <style>
    div[data-baseweb="select"] > div {
        padding-top: 2px;
        padding-bottom: 2px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# 🌍 Language and Navigation Bar (Same Row)
col1, col2 = st.columns([1, 3])  # Adjust width ratio as needed

with col1:
    target_lang = set_language_selector()

with col2:
    st.markdown(f"#### {translate('🔀 Navigate To', target_lang)}", unsafe_allow_html=True)
    tab_labels = [
        translate("🌿 Educational Content", target_lang),
        translate("👅 Tongue Health Check", target_lang),
        translate("🧠 Medical Review Dashboard", target_lang),
        translate("📊 TCM App Usage & Quality Dashboard", target_lang),
        translate("📚 About & Disclaimer", target_lang)
    ]
    selected_tab = st.selectbox("", tab_labels, key="tab_selector", label_visibility="collapsed")

# 🌟 Reset form when switching tabs
if "last_selected_tab" not in st.session_state:
    st.session_state.last_selected_tab = selected_tab

if selected_tab != st.session_state.last_selected_tab:
    for key in list(st.session_state.keys()):
        if key not in ["selected_lang", "language_selector", "last_selected_tab"]:
            del st.session_state[key]
    st.session_state.last_selected_tab = selected_tab

# ------------------------------
# EDUCATIONAL CONTENT 
# ------------------------------
if selected_tab == translate("🌿 Educational Content", target_lang):
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
elif selected_tab == translate("👅 Tongue Health Check", target_lang):
    st.title(translate("👅 Tongue Diagnosis Tool", target_lang))

    st.markdown(translate("Upload Tongue Image", target_lang))
    st.markdown(translate("Drag and drop a file below. Limit 200MB per file • JPG, JPEG, PNG", target_lang))
    # Ensure uploader runs only once
    if "uploaded_img" not in st.session_state:
        st.session_state.uploaded_img = None
    
    new_upload = st.file_uploader("", type=["jpg", "jpeg", "png"])

    # Update or clear session state
    if new_upload:
        st.session_state.uploaded_img = new_upload
    elif "uploaded_img" in st.session_state:
        st.session_state.pop("uploaded_img")  # Remove old image if none uploaded this time

    
    # Safely get final upload reference
    uploaded_img = st.session_state.get("uploaded_img", None)
    
    # Show preview only if the uploaded image is still valid
    if uploaded_img:
        try:
            uploaded_img.seek(0)
            img_bytes = uploaded_img.read()
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img, caption=translate("Preview of Uploaded Tongue Image", target_lang), width=300)
        except Exception as e:
            st.warning(translate("⚠️ Unable to preview uploaded image. Please re-upload." , target_lang))
            st.session_state.uploaded_img = None  # Clear invalid image


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
        uploaded_img = st.session_state.get("uploaded_img", None)
    
        if not uploaded_img or not consent:
            st.warning("Please upload image and give consent.")
            st.stop()
    
        submission_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
    
        uploaded_img.seek(0)  # ✅ CORRECT PLACEMENT here, not above
    
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
                st.markdown(f"**🩺 {translate('TCM Syndrome', target_lang)}:** {translate(gpt_response.get('tcm_syndrome', 'N/A'), target_lang)}")
                st.markdown(f"**💊 {translate('Western Equivalent', target_lang)}:** {translate(gpt_response.get('western_equivalent', 'N/A'), target_lang)}")
                
                st.markdown(f"**🌿 {translate('Remedies', target_lang)}:**")
                for r in gpt_response.get("remedies", []):
                    st.markdown(f"- {translate(r, target_lang)}")
                
                st.markdown(f"**⚖️ {translate('Discrepancies', target_lang)}:** {translate(gpt_response.get('discrepancies', 'N/A'), target_lang)}")
                st.markdown(f"**📊 {translate('Confidence Score', target_lang)}:** {gpt_response.get('confidence', 'N/A')}%")

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
elif selected_tab == translate("🧠 Medical Review Dashboard", target_lang):
    st.title(translate("🧠 Medical Review Dashboard", target_lang))
    st.info(translate("Select a submission to review and give expert feedback.", target_lang))
    
    docs = db.collection("gpt_diagnoses").stream()
    ids = [d.id for d in docs]
    selected_id = st.selectbox(translate("Submission ID", target_lang), ids)

    def show_table_side_by_side(expected_dict, actual_dict):
        st.write("### Comparison Table")
        table_data = []
        keys = sorted(set(expected_dict.keys()) | set(actual_dict.keys()))
        for key in keys:
            expected = expected_dict.get(key, "—")
            actual = actual_dict.get(key, "—")
            if isinstance(expected, list): expected = ", ".join(expected)
            if isinstance(actual, list): actual = ", ".join(actual)
            table_data.append([key, expected, actual])
        df = pd.DataFrame(table_data, columns=["Field", "User Input", "GPT Diagnosis"])
        styled_df = df.style.set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]}
        ]).set_properties(**{"text-align": "left"})
        st.table(styled_df)

    if selected_id:
        user_doc = db.collection("tongue_submissions").document(selected_id).get().to_dict()
        gpt_doc = db.collection("gpt_diagnoses").document(selected_id).get().to_dict()
        
        # Try fallback to gpt_doc if image_url is missing in user_doc
        image_url = None
        if user_doc and user_doc.get("image_url"):
            image_url = user_doc["image_url"]
            
        elif gpt_doc and gpt_doc.get("image_url"):
            image_url = gpt_doc["image_url"]
            st.caption("🧠 Image from `gpt_diagnoses`")

        
        # 📸 Tongue Image
        st.subheader("📸 Tongue Image")
        
        # Get image URL from either user_doc or gpt_doc
        image_url = None
        if user_doc and user_doc.get("image_url"):
            image_url = user_doc["image_url"]
            
        elif gpt_doc and gpt_doc.get("image_url"):
            image_url = gpt_doc["image_url"]
            st.caption("🧠 Image from `gpt_diagnoses`")
        
        if image_url:
            try:
                response = requests.get(image_url)
                content_type = response.headers.get("Content-Type", "")
                content_length = len(response.content)
                
        
                if response.status_code != 200:
                    st.error(f"❌ Failed to fetch image from Firebase. HTTP {response.status_code}")
                elif content_length == 0:
                    st.error("❌ The image file is empty (0 bytes). This likely means the upload failed.")
                elif "image" not in content_type:
                    st.error("❌ The returned file is not a valid image.")
                    st.code(image_url)
                else:
                    img = Image.open(BytesIO(response.content))
                    st.image(img, caption="Preview of Uploaded Tongue Image", width=300)
            except Exception as e:
                st.warning("⚠️ Failed to decode image.")
                st.exception(e)
        else:
            st.info("No image URL available.")

        

        # 📄 User Inputs
        user_inputs = user_doc.get("user_inputs", {})
        input_fields = {
            **{f"Symptom {i+1}": s for i, s in enumerate(user_inputs.get("symptoms", []))},
            **user_inputs.get("vitals", {}),
            **user_inputs.get("tongue_characteristics", {})
        }

        # 🤖 GPT Output
        raw_gpt = gpt_doc.get("gpt_response", "")
        gpt_data = {}
        
        # Detect fallback message or empty response
        fallback_phrases = [
            "i'm sorry", "i cannot", "unable to", "cannot analyze",
            "no valid", "please provide", "i can't", "not sure how"
        ]
        
        if isinstance(raw_gpt, dict):
            gpt_data = raw_gpt
        
        elif isinstance(raw_gpt, str):
            raw_trimmed = raw_gpt.strip().lower()
            if raw_trimmed == "" or any(p in raw_trimmed for p in fallback_phrases):
                st.warning("⚠️ GPT returned a fallback or non-actionable message.")
                st.code(raw_gpt if raw_gpt else "No data available", language="text")
            else:
                try:
                    parsed = json.loads(raw_gpt)
                    if isinstance(parsed, dict):
                        gpt_data = parsed
                    else:
                        st.warning("⚠️ Parsed GPT output is not a dictionary.")
                except json.JSONDecodeError as e:
                    st.warning(f"⚠️ Failed to parse GPT response as JSON: {e}")
                    st.code(raw_gpt, language="text")
        


        st.subheader("📊 User vs GPT-4o Comparison")
        if gpt_data:
            show_table_side_by_side(input_fields, gpt_data)
        elif isinstance(raw_gpt, str) and not gpt_data:
            # Avoid double message — fallback already shown above
            pass
        else:
            st.warning("⚠️ GPT response is not structured. Displaying raw fallback message:")
            st.code(raw_gpt if raw_gpt else "No data available", language="text")



        # 🧪 Optional Model Output
        model_doc = db.collection("model_outputs").document(selected_id).get().to_dict()
        if model_doc:
            st.subheader("🧪 Model Output (Internal)")
            st.json(model_doc.get("model_outputs", {}))
        else:
            st.info("No structured model output available.")

        # 🧬 Expert Feedback Form
        st.subheader("🧬 Expert Feedback")
        
        with st.form("expert_feedback_form"):
            agree = st.radio("Do you agree with the GPT diagnosis?", ["Yes", "Partially", "No"], key="agree_radio")
            corrected_syndrome = st.text_input("Correct TCM Syndrome", key="syndrome_input")
            corrected_equivalent = st.text_input("Correct Western Equivalent", key="equiv_input")
            corrected_remedies = st.text_area("Correct Remedies (comma-separated)", key="remedies_input")
            notes = st.text_area("Correction notes", key="notes_input")
        
            # ✅ Correct submit button inside the form
            submit_feedback = st.form_submit_button("📤 Submit Expert Feedback")
        
        if submit_feedback:
            feedback = {
                "submission_id": selected_id,
                "agreement": agree,
                "corrections": {
                    "tcm_syndrome": corrected_syndrome,
                    "western_equivalent": corrected_equivalent,
                    "remedies": [r.strip() for r in corrected_remedies.split(",") if r.strip()]
                },
                "notes": notes,
                "timestamp": datetime.utcnow().isoformat()
            }
            db.collection("medical_feedback").document(selected_id).set(feedback)
            st.success("✅ Feedback submitted successfully! Thank you for improving the model.")


            
# ------------------------------
# 📊 TCM App Usage & Quality Dashboard (Translation Ready)
# ------------------------------
elif selected_tab == translate("📊 TCM App Usage & Quality Dashboard", target_lang):
    st.title(translate("📊 TCM App Usage & Quality Dashboard", target_lang))
    
    try:
        # Load data from Firestore
        usage_docs = db.collection("tongue_submissions").stream()
        gpt_docs = db.collection("gpt_diagnoses").stream()
        feedback_docs = db.collection("medical_feedback").stream()
    
        usage_data = [doc.to_dict() for doc in usage_docs]
        gpt_data = [doc.to_dict() for doc in gpt_docs]
        feedback_data = [doc.to_dict() for doc in feedback_docs]
    
        df_usage = pd.DataFrame(usage_data)
        df_gpt = pd.DataFrame(gpt_data)
        df_fb = pd.DataFrame(feedback_data)
    
        # 🕒 Filter to last 30 days
        now = datetime.utcnow()
        last_30 = now - pd.Timedelta(days=30)
    
        df_usage["timestamp"] = pd.to_datetime(df_usage["timestamp"], errors="coerce")
        df_gpt["timestamp"] = pd.to_datetime(df_gpt["timestamp"], errors="coerce")
        df_fb["timestamp"] = pd.to_datetime(df_fb["timestamp"], errors="coerce")
    
        df_usage_recent = df_usage[df_usage["timestamp"] >= last_30]
        df_gpt_recent = df_gpt[df_gpt["timestamp"] >= last_30]
        df_fb_recent = df_fb[df_fb["timestamp"] >= last_30]
    
        # -------------------------
        # 📈 Top Metrics
        # -------------------------
        st.markdown(translate("### 📈 Key Metrics (Last 30 Days)", target_lang))
    
        col1, col2, col3 = st.columns(3)
    
        col1.metric(translate("🧪 Total Submissions", target_lang), len(df_usage_recent))
        col2.metric(translate("📝 Total Feedbacks", target_lang), len(df_fb_recent))
        agreement_rate = round(df_fb_recent["agreement"].eq("Yes").mean() * 100, 2) if not df_fb_recent.empty else 0
        col3.metric(translate("✅ Avg Agreement Rate", target_lang), f"{agreement_rate}%")
    
        st.markdown("---")
    
        # -------------------------
        # 📊 Submission Trend
        # -------------------------
        st.markdown(translate("### 📈 Submissions Over Time", target_lang))
    
        if not df_usage_recent.empty:
            submission_counts = df_usage_recent.groupby(df_usage_recent["timestamp"].dt.date).size()
            st.line_chart(submission_counts.rename(translate("Submissions", target_lang)))
        else:
            st.info(translate("No submissions in the last 30 days.", target_lang))
    
        # -------------------------
        # 📊 Expert Feedback Trend
        # -------------------------
        st.markdown(translate("### 🧬 Expert Feedback Agreement Over Time", target_lang))
    
        if not df_fb_recent.empty and "agreement" in df_fb_recent.columns:
            agreement_trend = df_fb_recent.groupby(df_fb_recent["timestamp"].dt.date)["agreement"].apply(
                lambda x: (x == "Yes").mean() * 100
            )
            st.line_chart(agreement_trend.rename(translate("% Agreement (Yes)", target_lang)))
        else:
            st.info(translate("No expert feedback collected yet.", target_lang))
    
        # -------------------------
        # 📥 Download Diagnoses
        # -------------------------
        st.markdown(translate("### 📥 Download All GPT Diagnoses", target_lang))
    
        if not df_gpt.empty:
            gpt_csv = df_gpt.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=translate("⬇️ Download GPT Diagnoses (CSV)", target_lang),
                data=gpt_csv,
                file_name="gpt_diagnoses.csv",
                mime="text/csv"
            )
        else:
            st.info(translate("No GPT diagnoses available for download.", target_lang))
    
    except Exception as e:
        st.error(translate("⚠️ Failed to load dashboard data.", target_lang))
        st.exception(e)


# ------------------------------
#  📚 ABOUT & DISCLAIMER (Prettier + Translated)
# ------------------------------
elif selected_tab == translate("📚 About & Disclaimer", target_lang):
    st.title(translate("📚 About Wise Tongue", target_lang))

    st.markdown(translate("""
        ### 🌿 What is Wise Tongue?
        
        **Wise Tongue** is your AI-powered Traditional Chinese Medicine (TCM) companion — blending ancient insights with modern AI technology.  
        This app aims to:
        
        - 📖 Educate users about the principles of TCM, especially tongue diagnosis.
        - 📸 Allow uploading of tongue images and symptom reports for AI-assisted wellness analysis.
        - 🧠 Leverage multimodal models (such as GPT-4o) to suggest TCM patterns, possible Western health analogies, and lifestyle recommendations.
        - 👩‍⚕️ Enable healthcare professionals to review, validate, and continuously improve AI-generated insights.
        
        ---
        
        ### 🔒 Data Usage & Privacy
        
        Your trust is important to us. Wise Tongue follows strict data handling practices:
        
        - 🔐 **Secure Storage:** All user data (images and self-reported symptoms) are securely stored in Firebase.
        - 🎯 **Purpose Limitation:** Data is used only for educational research, improving AI analysis, and studying global wellness trends.
        - 🕶️ **Anonymity:** Personal identities are **never collected**; all reviews are conducted anonymously.
        - 🩺 **Professional Feedback:** Medical professional feedback is used solely to refine and retrain AI models over time.
        
        ---
        
        ### ⚖️ Disclaimer
        
        Wise Tongue **does not provide medical diagnosis or treatment**.  
        It is intended for **educational, research, and wellness support purposes only**.
        
        Users should always consult licensed healthcare providers for any medical concerns or decisions.
        
        """, target_lang))
