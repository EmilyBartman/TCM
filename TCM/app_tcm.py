# -*- coding: utf-8 -*-
import streamlit as st
st.set_page_config(page_title="TCM Health App", layout="wide")

import pandas as pd
from PIL import Image
import os
from xhtml2pdf import pisa
from io import BytesIO
import base64
import uuid
from datetime import datetime, timedelta
import cv2
import numpy as np
import firebase_admin
from firebase_admin import storage, credentials, firestore
import joblib
from sklearn.ensemble import RandomForestClassifier
from deep_translator import GoogleTranslator
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from sklearn.linear_model import LogisticRegression
import torch
from torchvision import models, transforms



import torch
from torchvision import models, transforms

# Setup model
mobilenet = models.mobilenet_v3_small(pretrained=True)
mobilenet.classifier = torch.nn.Identity()
mobilenet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_features(cv_img):
    img_tensor = transform(cv_img).unsqueeze(0)
    features = mobilenet(img_tensor).squeeze().numpy()
    return features.tolist()


def load_model():
    model_path = "models/tcm_diagnosis_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None


def predict_with_model(model, features):
    try:
        st.write("🧪 Feature vector length:", len(features))
        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0].max()
        return pred, round(prob * 100, 2)
    except ValueError as e:
        st.error("❌ Model feature mismatch. Did you retrain?")
        st.exception(e)
        return "Model feature mismatch", 0




def retrain_model_from_firestore(db):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    import joblib

    docs = db.collection("tongue_features").stream()
    data = [doc.to_dict() for doc in docs if "features" in doc.to_dict() and "label" in doc.to_dict()]

    if not data:
        st.warning("❌ No labeled training data found.")
        return

    # Extract deep feature arrays and labels
    X = np.array([d["features"] for d in data])
    y = np.array([d["label"] for d in data])

    if X.shape[1] != 576:
        st.warning(f"⚠️ Cannot retrain: detected {X.shape[1]} features (expected 576).")
        return

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, "models/tcm_diagnosis_model.pkl")
    st.success("✅ Model retrained on deep features!")



def analyze_tongue_with_model(cv_img, submission_id, selected_symptoms, db):
    # UI-only average color (not for model anymore)
    avg_color = np.mean(cv_img.reshape(-1, 3), axis=0)
    avg_color_str = f"RGB({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})"

    # Extract deep features using PyTorch + MobileNetV3
    try:
        features = extract_features(cv_img)
    except Exception as e:
        st.error("❌ Failed to extract image features.")
        st.exception(e)
        return "Feature extraction error", "N/A", "N/A", [], 0

    # Load trained model
    model = load_model()

    if model:
        try:
            prediction_TCM, confidence = predict_with_model(model, features)
        except ValueError as e:
            st.warning("⚠️ Model feature mismatch – please retrain the model.")
            prediction_TCM, confidence = "Model feature mismatch", 0
    else:
        prediction_TCM, confidence = "Model not trained", 0

    prediction_Western = "N/A"

    # Store to Firebase
    if db:
        try:
            store_features_to_firestore(db, submission_id, features, prediction_TCM, confidence)
        except Exception as e:
            st.warning("⚠️ Could not store features to Firebase.")
            st.exception(e)

    return prediction_TCM, prediction_Western, avg_color_str, features, confidence


def store_features_to_firestore(db, submission_id, features, label, prob):
    features = [float(f) for f in features]

    db.collection("tongue_features").document(submission_id).set({
        "features": features,
        "label": str(label),
        "confidence": float(prob),
        "timestamp": firestore.SERVER_TIMESTAMP
    }, merge=True)
    
    db.collection("tongue_symptoms").document(submission_id).set({
    "features": features,
    "label": str(label),
    "confidence": float(prob),
    "symptom_count": len(selected_symptoms),
    "timestamp": firestore.SERVER_TIMESTAMP
}, merge=True)



def export_firestore_to_bigquery():
    pass  # Placeholder for actual implementation

def get_dynamic_remedies(tcm_pattern, symptoms=[]):
    remedies_map = {
        "Qi Deficiency": ["Ginseng tea", "Sweet potatoes", "Moderate walking"],
        "Yin Deficiency": ["Goji berries", "Pears & lily bulb soup", "Meditation"],
        "Blood Deficiency": ["Beets", "Spinach", "Dang Gui"],
        "Damp Retention": ["Barley water", "Avoid greasy food", "Pu-erh tea"]
    }
    return remedies_map.get(tcm_pattern, ["Balanced diet", "Hydration", "Rest"])



def render_dynamic_remedies(prediction_TCM, selected_symptoms):
    remedies = get_dynamic_remedies(prediction_TCM, selected_symptoms)
    st.markdown("**Suggestions:**")
    for item in remedies:
        st.markdown(f"- {item}")

# ---- FIREBASE SETUP ----
try:
    firebase_config = dict(st.secrets["firebase"])
    cred = credentials.Certificate(firebase_config)

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            "storageBucket": "traditional-medicine-50518"
        })

    db = firestore.client()
    bucket = storage.bucket("traditional-medicine-50518")
except Exception as e:
    db = None
    bucket = None
    st.error("❌ Firebase initialization failed")
    st.exception(e)

# ---- SESSION STATE ----
if "submissions" not in st.session_state:
    st.session_state.submissions = []

# ---- LANGUAGE SETUP ----
from deep_translator import GoogleTranslator
import streamlit as st

# -----------------------
# Language config
# -----------------------

languages = {
    "English": "en",
    "Spanish": "es",
    "Chinese (Simplified)": "zh-CN",  
    "Chinese (Traditional)": "zh-TW", 
    "French": "fr",
    "Hindi": "hi",
    "Arabic": "ar",
    "Swahili": "sw",
    "Zulu": "zu",
    "Amharic": "am",
    "Igbo": "ig",
    "Yoruba": "yo",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
    "Bengali": "bn",
    "Malay": "ms",
    "Vietnamese": "vi",
    "Thai": "th",
    "Filipino": "fil",
    "Japanese": "ja",
    "Korean": "ko"
}

# -----------------------
# Session state for language
# -----------------------

if "selected_lang" not in st.session_state:
    st.session_state.selected_lang = "English"

# Language selector (no callback)
new_lang = st.sidebar.selectbox(
    "🌐 Choose Language",
    list(languages.keys()),
    index=list(languages.keys()).index(st.session_state.selected_lang)
)

# Detect change and trigger rerun manually
if new_lang != st.session_state.selected_lang:
    st.session_state.selected_lang = new_lang
    st.rerun()

# Final resolved language code
target_lang = languages[st.session_state.selected_lang]

# -----------------------
# Translation helper
# -----------------------

def translate(text, lang_code):
    try:
        return GoogleTranslator(source="auto", target=lang_code).translate(text)
    except Exception as e:
        print(f"[Translation error] {e}")
        return text

# ---- NAVIGATION ----
pages = [
    "Educational Content",
    "Tongue Health Check",
    "Submission History",
    "About & Disclaimer"
]
page = st.sidebar.radio("Navigate", pages)


# ---- EDUCATIONAL CONTENT ----
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

# ---- TONGUE HEALTH CHECK ----

# ---- SUBMISSION HISTORY ----
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

# ---- TONGUE HEALTH CHECK ----
# ---- TONGUE HEALTH CHECK ----
elif page == "Tongue Health Check":
    st.title(translate("👅 Tongue Diagnosis Tool", target_lang))
    uploaded_img = st.file_uploader(translate("Upload your tongue image", target_lang), type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, caption=translate("Uploaded Image", target_lang), use_container_width=True)

    st.subheader(translate("Step 2: Describe Your Current Symptoms", target_lang))
    symptom_options = [
        "Fatigue", "Stress", "Stomach ache", "Headache", "Cold hands/feet",
        "Dry mouth", "Bloating", "Night sweats", "Nausea", "Dizziness"
    ]
    selected_symptoms = st.multiselect(translate("Select symptoms you are experiencing:", target_lang), symptom_options)

    st.subheader(translate("Step 3: Consent & Disclaimer", target_lang))
    consent = st.checkbox(translate("I consent to use of my image and data for research.", target_lang))
    st.info(translate("Not a medical diagnosis. For research and education only.", target_lang))

    if st.button(translate("🔍 Analyze My Tongue", target_lang)):
        if uploaded_img and consent:
            symptoms = ", ".join(selected_symptoms) if selected_symptoms else translate("None provided", target_lang)
            submission_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            file_ext = uploaded_img.name.split(".")[-1]
            firebase_filename = f"tongue_images/{submission_id}.{file_ext}"

            os.makedirs("temp", exist_ok=True)
            temp_path = f"temp/{submission_id}.{file_ext}"
            img.save(temp_path)

            cv_img = cv2.imread(temp_path)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            avg_color = np.mean(cv_img.reshape(-1, 3), axis=0)
            if avg_color[0] > 200 and avg_color[1] > 200 and avg_color[2] > 200:
                st.error(translate("⚠️ This image may not contain a tongue. Please upload a close-up of your tongue.", target_lang))
                st.stop()

            prediction_TCM, prediction_Western, avg_color_str, features, confidence = analyze_tongue_with_model(
                cv_img, submission_id, selected_symptoms, db
            )

            try:
                blob = bucket.blob(firebase_filename)
                blob.upload_from_filename(temp_path)
                url = blob.generate_signed_url(expiration=timedelta(hours=1), method="GET")
                img_url = url
                st.success(translate("✅ Image uploaded.", target_lang))
                st.markdown(f"🔗 [{translate('View Uploaded Image', target_lang)}]({img_url})")
            except Exception as e:
                st.error(translate("❌ Upload to Firebase failed.", target_lang))
                st.exception(e)
                st.stop()

            result = {
                "id": submission_id,
                "timestamp": timestamp,
                "symptoms": symptoms,
                "tongue_image_url": img_url,
                "avg_color": avg_color_str,
                "prediction_TCM": prediction_TCM,
                "prediction_Western": prediction_Western,
                "user_feedback": ""
            }
            st.session_state.submissions.append(result)
            db.collection("tongue_scans").document(submission_id).set(result)
            st.subheader(translate("🧪 Analysis Results", target_lang))
            st.info(f"🔍 {translate('Detected TCM Pattern:', target_lang)} **{prediction_TCM}** | {translate('Western View:', target_lang)} **{prediction_Western}**")         
            
            # Extract RGB values from the avg_color_str
            r, g, b = map(int, avg_color_str.strip("RGB()").split(","))
            hex_color = '#%02x%02x%02x' % (r, g, b)
            
            # Display translated label + value + color swatch
            avg_color = np.mean(cv_img.reshape(-1, 3), axis=0)
            r, g, b = map(int, avg_color)
            hex_color = '#%02x%02x%02x' % (r, g, b)
            avg_color_str = f"RGB({r}, {g}, {b})"

            st.markdown(f"**{translate('Confidence Level', target_lang)}**: `{confidence}%`")

            render_dynamic_remedies(prediction_TCM, selected_symptoms)

            st.markdown("- " + translate("Hydration, nutrition, and better sleep often help.", target_lang))


           # --- More Details Suggested Remedies ---
            with st.expander(translate("🌿 More Details on Suggested Remedies Based on TCM Pattern", target_lang)):
                remedy_text = ""
            
                if prediction_TCM == "Qi Deficiency":
                    remedy_text = """
            ✅ Ginseng tea - An adaptogenic herbal tonic that boosts Qi, supports immune function, and improves stamina.  
            🍠 Sweet potatoes - Nutrient-rich root vegetable that strengthens the spleen and digestion, a key organ in Qi production.  
            🚶‍♂️ Moderate exercise like walking - Gentle physical movement like walking or Tai Chi stimulates Qi flow, circulation, and reduces fatigue.
                    """
                elif prediction_TCM == "Yin Deficiency":
                    remedy_text = """
            🍒 Goji berries - A Yin-nourishing superfruit traditionally used to support the liver, eyes, and immune system.  
            🍐 Pears and lily bulb soup - A classic TCM remedy that clears heat and nourishes Yin.  
            🧘 Meditation and rest - Meditation calms excessive Yang, reduces stress, and restores Yin energy.
                    """
                elif prediction_TCM == "Blood Deficiency":
                    remedy_text = """
            🥬 Beets, spinach, black beans - Iron-rich, blood-nourishing foods that replenish energy and nutrients.  
            🌿 Dang Gui (Angelica Sinensis) - A powerful herb for enriching and circulating blood.  
            🩸 Iron-rich foods - Support red blood cell production and help address cold extremities and pale complexion.
                    """
                elif prediction_TCM == "Damp Retention":
                    remedy_text = """
            🥣 Barley water - Cooling drink that helps remove excess dampness and reduce bloating.  
            🚫 Avoid greasy food - Oily meals create damp buildup and sluggish digestion.  
            🍵 Ginger and pu-erh tea - Warming teas that reduce phlegm and support healthy digestion.
                    """
                else:
                    remedy_text = """
            💧 Maintain hydration - Drink enough water to support detox and overall health.  
            🥗 Balanced meals - Whole foods build Qi and support organ function.  
            🧘 Gentle exercise - Yoga, stretching, or walking helps reduce stress and improve flow.
                    """
            
                st.markdown(translate(remedy_text.strip(), target_lang))
            

            # --- PDF Download ---
            with st.expander(translate("📄 Download Report", target_lang)):
                html_report = f"""
                <h2>{translate('TCM Health Scan Report', target_lang)}</h2>
                <p><strong>{translate('Timestamp', target_lang)}:</strong> {timestamp}</p>
                <p><strong>{translate('Symptoms', target_lang)}:</strong> {symptoms}</p>
                <p><strong>{translate('Color', target_lang)}:</strong> {avg_color_str} - {prediction_TCM}</p>
                <p><strong>{translate('Western Insight', target_lang)}:</strong> {prediction_Western}</p>
                <p><strong>{translate('Confidence', target_lang)}:</strong> {confidence}%</p>
                """
                pdf_output = BytesIO()
                pisa.CreatePDF(BytesIO(html_report.encode("utf-8")), dest=pdf_output)
                pdf_bytes = pdf_output.getvalue()
                b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                download_link = f'<a href="data:application/pdf;base64,{b64}" download="tcm_report.pdf">📥 {translate("Download PDF Report", target_lang)}</a>'
                st.markdown(download_link, unsafe_allow_html=True)

            # --- History Compare ---
            with st.expander(translate("📊 Compare with Previous Scans", target_lang)):
                scans = db.collection("tongue_scans").where("id", "!=", submission_id).stream()
                history = [doc.to_dict() for doc in scans if doc.to_dict().get("prediction_TCM")]
                if history:
                    hist_df = pd.DataFrame(history)
                    hist_df = hist_df.sort_values("timestamp", ascending=False).head(5)
                    st.dataframe(hist_df[["timestamp", "prediction_TCM", "prediction_Western"]])
                else:
                    st.write(translate("No prior scans available to compare.", target_lang))

            # --- Feedback Section ---
            feedback = st.radio(
                translate("Was this prediction accurate?", target_lang),
                [translate("Not sure", target_lang), translate("Yes", target_lang), translate("No", target_lang)],
                index=0
            )
            if st.button(translate("Submit Feedback", target_lang)):
                if feedback in [translate("Yes", target_lang), translate("No", target_lang)]:
                    db.collection("tongue_features").document(submission_id).update({
                        "is_correct": True if feedback == translate("Yes", target_lang) else False
                    })
                    st.toast(translate("Feedback submitted. Thank you!", target_lang), icon="\U0001F4EC")

                else:
                    st.warning(translate("Please select 'Yes' or 'No' to submit feedback.", target_lang))

            if st.button("🔁 Retrain Model"):
                retrain_model_from_firestore(db)

                    
# ---- SUBMISSION HISTORY ----
elif page == "Submission History":
    st.title(translate("📜 My Tongue Scan History", target_lang))
    if st.session_state.submissions:
        df = pd.DataFrame(st.session_state.submissions)
        st.dataframe(df.rename(columns={
            "timestamp": translate("Timestamp", target_lang),
            "prediction_TCM": translate("TCM Prediction", target_lang),
            "prediction_Western": translate("Western Insight", target_lang)
        }))

        if "is_correct" in df.columns:
            correct_count = df["is_correct"].sum()
            total_feedback = df["is_correct"].notna().sum()
            if total_feedback > 0:
                accuracy = round((correct_count / total_feedback) * 100, 2)
                st.metric(translate("📊 Model Accuracy (based on feedback)", target_lang), f"{accuracy}%")

            st.subheader(translate("📈 Accuracy Over Time", target_lang))
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            daily = df[df["is_correct"].notna()].groupby(df["timestamp"].dt.date)["is_correct"].mean()
            st.line_chart(daily)

            st.subheader(translate("🧪 Accuracy by TCM Syndrome", target_lang))
            if "prediction_TCM" in df.columns:
                by_syndrome = df[df["is_correct"].notna()].groupby("prediction_TCM")["is_correct"].mean().sort_values(ascending=False)
                st.bar_chart(by_syndrome.rename(index=lambda x: translate(x, target_lang)))

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(translate("⬇️ Download CSV", target_lang), csv, "my_tongue_scans.csv", "text/csv")
    else:
        st.info(translate("You haven't submitted any scans yet.", target_lang))

# ---- ABOUT & DISCLAIMER ----
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
