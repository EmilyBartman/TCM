# -*- coding: utf-8 -*-
import streamlit as st
st.set_page_config(page_title="TCM Health App", layout="wide")

import pandas as pd
import numpy as np
import cv2
import os
import uuid
import base64
from PIL import Image
from datetime import datetime, timedelta
from io import BytesIO
from xhtml2pdf import pisa
from collections import Counter

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

import torch
from torchvision import models, transforms
from sklearn.ensemble import RandomForestClassifier
import joblib

import firebase_admin
from firebase_admin import credentials, firestore, storage

from deep_translator import GoogleTranslator

# in Tongue_Health_Page.py
from shared_utils import ensure_model_loaded
# in app_tcm.py
from Tongue_Health_Page import render_tongue_health_check


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

# ---- MODEL SETUP ----
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

# ---- MODEL MANAGEMENT ----

def ensure_model_loaded():
    if "tcm_model" not in st.session_state or st.session_state.tcm_model is None:
        model_path = "models/tcm_diagnosis_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.session_state.tcm_model = model

def load_model():
    model_path = "models/tcm_diagnosis_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.session_state.tcm_model = model  # ✅ refresh session cache
        return model
    else:
        st.warning("⚠️ No model file found. Automatically retraining using Firestore data...")
        retrain_model_from_firestore(get_firestore_client())

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.session_state.tcm_model = model  # ✅ set again after retrain
            return model
        else:
            st.error("❌ Retrain failed. No model file saved.")
            return None


def retrain_model_from_firestore(db):
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    st.info("🧪 Fetching training data from Firestore...")
    docs = db.collection("tongue_features").stream()
    raw = [doc.to_dict() for doc in docs if "features" in doc.to_dict() and "label" in doc.to_dict()]

    st.write(f"📦 Total docs in Firestore: {len(raw)}")

    filtered = [d for d in raw if isinstance(d["features"], list)]
    st.write(f"🧪 Documents with list-type features: {len(filtered)}")

    valid = [d for d in filtered if len(d["features"]) == 576]
    st.write(f"✅ Valid training samples (576 features): {len(valid)}")

    if not valid:
        st.error("❌ No valid training samples found with 576 features.")
        return

    X = np.array([d["features"] for d in valid])
    y = np.array([d["label"] for d in valid])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    model_path = "models/tcm_diagnosis_model.pkl"

    try:
        joblib.dump(model, model_path)
        st.success(f"✅ Model saved to: `{model_path}`")
        st.session_state.tcm_model = model  # ✅ Ensure it's loaded into session
    except Exception as e:
        st.error("❌ Failed to save model.")
        st.exception(e)


def predict_with_model(model, features):
    try:
        st.write("🧪 Feature vector length:", len(features))
        pred = model.predict([features])[0]
        prob_array = model.predict_proba([features])[0]
        prob = prob_array.max()
        st.write(f"🧬 Raw Prediction: {pred}")
        st.write(f"📈 Probabilities: {prob_array}")
        return pred, round(prob * 100, 2)
    except ValueError as e:
        st.error("❌ Model feature mismatch.")
        st.exception(e)
        return "Model feature mismatch", 0

def analyze_tongue_with_model(cv_img, submission_id, selected_symptoms, db):
    avg_color = np.mean(cv_img.reshape(-1, 3), axis=0)
    avg_color_str = f"RGB({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})"

    try:
        features = extract_features(cv_img)
        assert len(features) == 576
    except Exception as e:
        st.error("❌ Failed to extract image features.")
        st.exception(e)
        return "Feature extraction error", "N/A", avg_color_str, [], 0

    model = st.session_state.get("tcm_model")
    st.code(f"🔍 Model present: {bool(model)} | Features: {len(features)}")
    prediction_TCM, confidence = "Model not trained", 0

    if model:
        try:
            prediction_TCM, confidence = predict_with_model(model, features)
        except Exception as e:
            st.warning("⚠️ Error during prediction.")
            st.exception(e)
            prediction_TCM, confidence = "Prediction error", 0
    else:
        st.error("⚠️ Model not available in session. You may need to reload or retrain.")

    prediction_Western = "N/A"

    # Only store if features valid and not default message
    if db and prediction_TCM not in ["Model not trained", "Prediction error"]:
        try:
            store_features_to_firestore(db, submission_id, features, prediction_TCM, confidence, selected_symptoms)
        except Exception as e:
            st.warning("⚠️ Could not store features to Firebase.")
            st.exception(e)

    st.code(f"Prediction: {prediction_TCM} | Confidence: {confidence}")
    return prediction_TCM, prediction_Western, avg_color_str, features, confidence

def store_features_to_firestore(db, submission_id, features, label, prob, selected_symptoms):
    features = [float(f) for f in features]
    if len(features) == 576:
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
    else:
        st.warning(f"⚠️ Not storing to Firestore. Feature length is {len(features)}, expected 576.")

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
# ---- SESSION STATE ----
if "submissions" not in st.session_state:
    st.session_state.submissions = []

if "selected_lang" not in st.session_state:
    st.session_state.selected_lang = "English"

# ---- LANGUAGE SETUP ----
languages = {
    "English": "en", "Spanish": "es", "Chinese (Simplified)": "zh-CN",  
    "Chinese (Traditional)": "zh-TW", "French": "fr", "Hindi": "hi",
    "Arabic": "ar", "Swahili": "sw", "Zulu": "zu", "Amharic": "am",
    "Igbo": "ig", "Yoruba": "yo", "Tamil": "ta", "Telugu": "te",
    "Urdu": "ur", "Bengali": "bn", "Malay": "ms", "Vietnamese": "vi",
    "Thai": "th", "Filipino": "fil", "Japanese": "ja", "Korean": "ko"
}

# ---- LANGUAGE SELECTOR ----
new_lang = st.sidebar.selectbox(
    "🌐 Choose Language",
    list(languages.keys()),
    index=list(languages.keys()).index(st.session_state.selected_lang)
)

if new_lang != st.session_state.selected_lang:
    st.session_state.selected_lang = new_lang
    st.rerun()

target_lang = languages[st.session_state.selected_lang]

# ---- TRANSLATION HELPER ----
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
In TCM, the tongue reflects internal health: its color, coating, and moisture can reveal organ imbalance.

**Examples**:
- Pale: Qi/Blood deficiency
- Yellow coat: Heat or damp-heat
- Cracks: Yin deficiency
- Purple: Blood stasis
""", target_lang))

    st.header(translate("🌐 Bridging TCM and Western Medicine", target_lang))
    st.markdown(translate("""
| Concept | TCM View | Western Analogy |
|--------|----------|----------------|
| Qi      | Energy flow | Nervous/Circulatory function |
| Yin/Yang| Balance | Hormonal/homeostasis |
| Tongue Diagnosis | Internal imbalance | Anemia, dehydration, oral signs |
""", target_lang))

# ---- TONGUE HEALTH CHECK PAGE ----
elif page == "Tongue Health Check":
    from Tongue_Health_Page import render_tongue_health_check
    render_tongue_health_check(
        analyze_tongue_with_model,
        db=db,
        bucket=bucket,
        translate=translate,
        target_lang=target_lang
    )


# ---- SUBMISSION HISTORY ----
elif page == "Submission History":
    st.title(translate("📜 My Tongue Scan History", target_lang))

    if st.session_state.submissions:
        df = pd.DataFrame(st.session_state.submissions)
        df_display = df.rename(columns={
            "timestamp": translate("Timestamp", target_lang),
            "prediction_TCM": translate("TCM Pattern", target_lang),
            "prediction_Western": translate("Western View", target_lang)
        })
        st.dataframe(df_display)

        if "is_correct" in df.columns:
            correct = df["is_correct"].sum()
            total = df["is_correct"].notna().sum()
            if total > 0:
                acc = round((correct / total) * 100, 2)
                st.metric(translate("Model Accuracy", target_lang), f"{acc}%")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(translate("⬇️ Download CSV", target_lang), csv, "tongue_history.csv", "text/csv")
    else:
        st.info(translate("You haven't submitted any scans yet.", target_lang))

# ---- ABOUT & DISCLAIMER ----
elif page == "About & Disclaimer":
    st.title(translate("About This App", target_lang))
    st.markdown(translate("""
This app helps demonstrate how AI can analyze tongue health for TCM research.

🔬 AI Model: Uses deep image features and Random Forest
🧠 Purpose: Research + Education, not clinical use
📁 Data: Stored securely and anonymously
🧾 Feedback: Helps retrain and improve predictions

⚠️ This is NOT a substitute for professional medical advice.
""", target_lang))

