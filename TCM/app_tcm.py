# -*- coding: utf-8 -*-
"""app_TCM.py
Automatically generated by Colab.
Original file is located at
    https://colab.research.google.com/drive/1tJWVfjSj5I-wK474ZfG83HN2mb5Gvdmf
"""

# Traditional Chinese Medicine Web App MVP (Streamlit)
import streamlit as st
import pandas as pd
from PIL import Image
import os
import uuid
from datetime import datetime
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials
from google.oauth2 import service_account
from google.cloud import storage as gcs_storage

st.write("✅ App started loading")

# ---- FIREBASE & GCS SETUP ----
firebase_config = dict(st.secrets["firebase"])
cred = credentials.Certificate(firebase_config)

# Firebase Admin init
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

st.write("✅ Firebase config loaded")

# TEMP DISABLE FIRESTORE
db = None
st.write("⚠️ Firestore skipped for now")

# Skipping GCS init temporarily
st.write("✅ Skipped GCS init temporarily")

# ---- STREAMLIT SETUP ----
st.set_page_config(page_title="TCM Health App", layout="wide")
if "submissions" not in st.session_state:
    st.session_state.submissions = []

# ---- NAVIGATION ----
try:
    pages = ["Educational Content", "Tongue Health Check", "About & Disclaimer"]
    page = st.sidebar.radio("Navigate", pages)
    st.write(f"✅ Current page: {page}")
except Exception as e:
    st.error("❌ Sidebar navigation failed")
    st.exception(e)

# ---- 1. EDUCATIONAL CONTENT ----
if page == "Educational Content":
    st.title("🌿 Traditional Chinese Medicine (TCM) Basics")
    st.header("What is TCM?")
    st.write("""
        Traditional Chinese Medicine is a holistic approach to health that includes practices like acupuncture,
        herbal medicine, massage (Tui Na), exercise (Qi Gong), and dietary therapy. It is based on concepts such as:
        - **Yin & Yang**: Balance between opposite but complementary forces
        - **Qi (Chi)**: Vital energy flowing through the body
        - **Five Elements**: Wood, Fire, Earth, Metal, Water—linked to organs and emotions

        TCM often contrasts with **Western medicine**, which tends to focus on pathology, lab diagnostics, and medications.
    """)

    st.subheader("Sources")
    st.markdown("- [WHO on TCM](https://www.who.int/health-topics/traditional-complementary-and-integrative-medicine)")
    st.markdown("- [PubMed on TCM Research](https://pubmed.ncbi.nlm.nih.gov/?term=traditional+chinese+medicine)")

# ---- 2. TONGUE HEALTH CHECK ----
elif page == "Tongue Health Check":
    st.title("👅 Tongue Diagnosis Tool")

    st.subheader("Step 1: Upload Your Tongue Image")
    uploaded_img = st.file_uploader("Upload a clear image of your tongue", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded Tongue Image", use_container_width=True)

    st.subheader("Step 2: Describe Your Current Symptoms")
    symptoms = st.text_area("Describe what you’re feeling physically or emotionally (e.g. tiredness, stress, stomach ache)")

    st.subheader("Step 3: Consent & Disclaimer")
    consent = st.checkbox("I consent to the use of my image and data for future research and model training.")

    st.info("This app does not provide a medical diagnosis. It is for educational and research purposes only.")

    st.subheader("Step 4: Submit for Analysis")
    if st.button("🔍 Analyze My Tongue"):
        if uploaded_img and consent:
            submission_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            file_ext = uploaded_img.name.split(".")[-1]
            firebase_filename = f"tongue_images/{submission_id}.{file_ext}"

            # Save image temporarily for analysis
            os.makedirs("temp", exist_ok=True)
            temp_path = f"temp/{submission_id}.{file_ext}"
            img.save(temp_path)

            # ----- BASIC IMAGE ANALYSIS -----
            cv_img = cv2.imread(temp_path)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(cv_img, (300, 300))

            avg_color = np.mean(resized.reshape(-1, 3), axis=0)
            avg_color_str = f"RGB({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})"

            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            shape_comment = "Normal" if edge_pixels < 5000 else "Swollen or Elongated"
            texture_comment = "Moist" if laplacian_var < 100 else "Dry/Coated"

            # Simulate upload to GCS
            try:
                img_url = "https://storage.googleapis.com/demo-placeholder.png"
                st.success("✅ Simulated image upload.")
                st.write(f"🔗 Simulated Public URL: {img_url}")
            except Exception as e:
                st.error("❌ Upload to Firebase failed.")
                st.exception(e)
                st.stop()

            # Store metadata (skipped)
            data_row = {
                "id": submission_id,
                "timestamp": timestamp,
                "symptoms": symptoms,
                "tongue_image_url": img_url,
                "avg_color": avg_color_str,
                "shape_comment": shape_comment,
                "texture_comment": texture_comment,
                "prediction_TCM": "Qi Deficiency (placeholder)",
                "prediction_Western": "Possible Fatigue/Anemia (placeholder)",
                "user_feedback": ""
            }
            # db.collection("tongue_scans").document(submission_id).set(data_row)

            st.success("Image submitted and analyzed successfully! Scroll down for prediction.")
            st.write(f"Connected to Firestore (disabled)")

            # Display prediction & analysis
            st.subheader("🧪 Analysis Results")
            st.markdown(f"- **Tongue Color**: {avg_color_str}")
            st.markdown(f"- **Shape Analysis**: {shape_comment}")
            st.markdown(f"- **Coating / Moisture Level**: {texture_comment}")
            st.markdown("- **TCM Insight**: Qi Deficiency (based on image features)")
            st.markdown("- **Western Equivalent**: Possible signs of fatigue or low hemoglobin")

            # Feedback (skipped)
            st.subheader("How accurate was this?")
            feedback = st.text_input("Your feedback or correction (optional)")
            if feedback:
                st.success("Thanks! Your feedback helps improve our model.")
                # db.collection("tongue_scans").document(submission_id).update({"user_feedback": feedback})
        else:
            st.error("Please upload an image and provide consent.")

# ---- 3. ABOUT & DISCLAIMER ----
elif page == "About & Disclaimer":
    st.title("📜 About This App")
    st.write("""
        This application is a prototype designed to:
        - Educate the public on Traditional Chinese Medicine
        - Demonstrate how tongue analysis can indicate overall health
        - Begin building a dataset for research into health diagnostics via images

        ⚠️ **Disclaimer**: This app does NOT replace professional medical advice.
        Your image and responses will be stored securely and anonymously, used only for improving predictive capabilities.
    """)
