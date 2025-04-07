# -*- coding: utf-8 -*-
import streamlit as st
st.set_page_config(page_title="TCM Health App", layout="wide")

import pandas as pd
from PIL import Image
import os
import uuid
from datetime import datetime, timedelta
import cv2
import numpy as np
import firebase_admin
from firebase_admin import storage, credentials, firestore

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
    # st.success("✅ Firebase and Firestore initialized")  # Hidden from UI
except Exception as e:
    db = None
    bucket = None
    st.error("❌ Firebase initialization failed")
    st.exception(e)

# ---- SESSION STATE ----
if "submissions" not in st.session_state:
    st.session_state.submissions = []

pages = [
    "Educational Content",
    "Tongue Health Check",
    "Submission History",
    "About & Disclaimer"
]
page = st.sidebar.radio("Navigate", pages)

# ---- EDUCATIONAL CONTENT ----
if page == "Educational Content":
    st.title("🌿 Traditional Chinese Medicine (TCM) Education")
    st.header("Foundations of TCM")
    st.markdown("""
    - **Yin & Yang**: Balance of opposing but complementary forces.
    - **Qi (Chi)**: Vital life energy.
    - **Five Elements**: Wood, Fire, Earth, Metal, Water—linked to organs/emotions.
    - **Diagnostic Tools**: Pulse, tongue, face, symptom observation.
    - **Modalities**: Acupuncture, herbal therapy, dietary therapy, Qi Gong.
    """)

    st.header("🔎 Why the Tongue Matters in TCM")
    st.markdown("""
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
    """)

    st.header("🌐 Bridging TCM and Western Medicine")
    st.markdown("""
    Traditional Chinese Medicine (TCM) and Western medicine differ in philosophy and methods but can be complementary:

    | Concept | TCM Interpretation | Western Medicine Analogy |
    |--------|---------------------|---------------------------|
    | Qi (Vital Energy) | Flow of life energy through meridians | Nervous & Circulatory System activity |
    | Yin/Yang | Balance of cold/hot, passive/active forces | Homeostasis (e.g., hormonal balance) |
    | Tongue Diagnosis | Reflects internal organ status | Inflammation markers, dehydration, anemia |
    | Syndrome Differentiation | Pattern-based holistic assessment | Evidence-based diagnosis (labs, scans) |

    Integrative medicine combines both paradigms to enhance wellness, prevention, and personalized care.
    """)

    st.header("TCM Syndrome Library")
    with st.expander("🔎 Click to view 8 Major Tongue Syndromes and Signs"):
        st.markdown("""
        **Qi Deficiency**: Fatigue, pale tongue, short breath
        **Damp Retention**: Bloating, sticky tongue coat
        **Blood Stasis**: Sharp pain, purple tongue
        **Qi Stagnation**: Emotional blockage, rib pain
        **Damp Heat**: Yellow tongue coat, foul smell
        **Yang Deficiency**: Cold limbs, low energy
        **Yin Deficiency**: Dry mouth, night sweats
        **Blood Deficiency**: Pale lips, dizziness
        """)
    with st.expander("📚 Recommended Reading"):
        st.markdown("""
        - *Foundations of Chinese Medicine* - Giovanni Maciocia
        - *Healing with Whole Foods* - Paul Pitchford
        - *The Web That Has No Weaver* - Ted J. Kaptchuk
        - [WHO on TCM](https://www.who.int/health-topics/traditional-complementary-and-integrative-medicine)
        - [PubMed on TCM Research](https://pubmed.ncbi.nlm.nih.gov/?term=traditional+chinese+medicine)
        """)

# ---- TONGUE HEALTH CHECK ----
elif page == "Tongue Health Check":
    st.title("👅 Tongue Diagnosis Tool")
    uploaded_img = st.file_uploader("Upload your tongue image", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded Image", use_container_width=True)

    st.subheader("Step 2: Describe Your Current Symptoms")
    symptom_options = [
        "Fatigue", "Stress", "Stomach ache", "Headache", "Cold hands/feet",
        "Dry mouth", "Bloating", "Night sweats", "Nausea", "Dizziness"
    ]
    selected_symptoms = st.multiselect("Select symptoms you are experiencing:", symptom_options)

    st.subheader("Step 3: Consent & Disclaimer")
    consent = st.checkbox("I consent to use of my image and data for research.")
    st.info("Not a medical diagnosis. For research and education only.")

    if st.button("🔍 Analyze My Tongue"):
        if uploaded_img and consent:
            symptoms = ", ".join(selected_symptoms) if selected_symptoms else "None provided"
            # [rest of analysis logic remains unchanged]
        else:
            st.error("❌ Please upload an image and agree to consent.")

# ---- SUBMISSION HISTORY ----
elif page == "Submission History":
    st.title("📜 My Tongue Scan History")
    if st.session_state.submissions:
        df = pd.DataFrame(st.session_state.submissions)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "my_tongue_scans.csv", "text/csv")
    else:
        st.info("You haven't submitted any scans yet.")

# ---- ABOUT & DISCLAIMER ----
elif page == "About & Disclaimer":
    st.title("ℹ️ About This App")
    st.markdown("""
    This app is built for:
    - Educating users about TCM tongue diagnostics
    - Demonstrating how AI may assist in early wellness screening
    - Researching global health variations using tongue + symptom data

    🔒 **Data Usage**: All uploaded data is securely stored and used anonymously for improving model prediction.

    ⚠️ **Disclaimer**: This tool is for educational purposes only. It does not replace medical diagnosis or professional care.
    """)
