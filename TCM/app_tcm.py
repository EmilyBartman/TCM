# -*- coding: utf-8 -*-
"""TCM Web App - Streamlit MVP"""
import streamlit as st
from PIL import Image
import os
import uuid
from datetime import datetime
import cv2
import json
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore, storage as fb_storage

# --- PAGE CONFIG ---
st.set_page_config(page_title="TCM Health App", layout="wide")

# --- SESSION INIT ---
if "submissions" not in st.session_state:
    st.session_state.submissions = []
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Educational Content"

st.write("✅ App started loading")

# --- FIREBASE SETUP ---
try:
    firebase_config = dict(st.secrets["firebase"])

    # Write config to a temporary file
    os.makedirs("secrets", exist_ok=True)
    with open("secrets/firebase_temp.json", "w") as f:
        json.dump(firebase_config, f)

    cred = credentials.Certificate("secrets/firebase_temp.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            "storageBucket": f"{firebase_config['project_id']}.appspot.com"
        })

    db = firestore.client()
    bucket = fb_storage.bucket()
    st.write("✅ Firebase initialized successfully")
except Exception as e:
    db = None
    bucket = None
    st.error("❌ Firebase initialization failed")
    st.exception(e)

# --- SIDEBAR NAVIGATION ---
try:
    pages = ["Educational Content", "Tongue Health Check", "About & Disclaimer"]
    selected_index = pages.index(st.session_state.get("selected_page", "Educational Content"))
    page = st.sidebar.radio("Navigate", pages, index=selected_index)
    st.session_state.selected_page = page
    st.write(f"✅ Current page: {page}")
    st.write("📄 Debug: Session state", st.session_state)
except Exception as e:
    st.error("❌ Sidebar navigation failed")
    st.exception(e)

# --- PAGE CONTENT RENDERING ---
try:
    if page == "Educational Content":
        st.title("🌿 Traditional Chinese Medicine (TCM) Basics")
        st.header("What is TCM?")
        st.write("""
            Traditional Chinese Medicine is a holistic approach to health including:
            - **Yin & Yang**: Balance between opposite but complementary forces
            - **Qi (Chi)**: Vital energy flowing through the body
            - **Five Elements**: Wood, Fire, Earth, Metal, Water—linked to organs and emotions

            TCM often contrasts with **Western medicine**, which focuses on pathology and medication.
        """)
        st.subheader("Sources")
        st.markdown("- [WHO on TCM](https://www.who.int/health-topics/traditional-complementary-and-integrative-medicine)")
        st.markdown("- [PubMed on TCM Research](https://pubmed.ncbi.nlm.nih.gov/?term=traditional+chinese+medicine)")

    elif page == "Tongue Health Check":
        st.title("👅 Tongue Diagnosis Tool")

        uploaded_img = st.file_uploader("Upload a clear image of your tongue", type=["jpg", "jpeg", "png"])
        if uploaded_img:
            img = Image.open(uploaded_img)
            st.image(img, caption="Uploaded Tongue Image", use_container_width=True)

        symptoms = st.text_area("Describe your symptoms (e.g. tiredness, stress, stomach ache)")
        consent = st.checkbox("I consent to use of my image and data for research and model training.")
        st.info("This app does not provide medical diagnoses. For educational use only.")

        if st.button("🔍 Analyze My Tongue"):
            if uploaded_img and consent:
                submission_id = str(uuid.uuid4())
                timestamp = datetime.utcnow().isoformat()
                file_ext = uploaded_img.name.split(".")[-1]
                filename = f"{submission_id}.{file_ext}"
                os.makedirs("temp", exist_ok=True)
                temp_path = os.path.join("temp", filename)
                img.save(temp_path)

                # Process image
                cv_img = cv2.cvtColor(cv2.imread(temp_path), cv2.COLOR_BGR2RGB)
                resized = cv2.resize(cv_img, (300, 300))
                avg_color = np.mean(resized.reshape(-1, 3), axis=0)
                avg_color_str = f"RGB({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})"
                gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_pixels = np.sum(edges > 0)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                shape_comment = "Normal" if edge_pixels < 5000 else "Swollen or Elongated"
                texture_comment = "Moist" if laplacian_var < 100 else "Dry/Coated"

                # Upload to Firebase Storage
                if bucket:
                    blob = bucket.blob(f"tongue_images/{filename}")
                    blob.upload_from_filename(temp_path)
                    blob.make_public()
                    img_url = blob.public_url
                    st.success("✅ Image uploaded to Firebase Storage")
                else:
                    img_url = "https://storage.googleapis.com/demo-placeholder.png"
                    st.warning("Firebase Storage not available. Using placeholder URL.")

                data = {
                    "id": submission_id,
                    "timestamp": timestamp,
                    "symptoms": symptoms,
                    "image_url": img_url,
                    "avg_color": avg_color_str,
                    "shape_comment": shape_comment,
                    "texture_comment": texture_comment,
                    "prediction_TCM": "Qi Deficiency (placeholder)",
                    "prediction_Western": "Possible Fatigue/Anemia (placeholder)"
                }

                if db:
                    db.collection("tongue_diagnostics").document(submission_id).set(data)
                    st.success("✅ Submission saved to Firestore")
                else:
                    st.warning("Firestore not available. Data not saved to database.")

                st.session_state.submissions.append(data)

                # Show results
                st.subheader("🧪 Analysis Results")
                st.markdown(f"- **Tongue Color**: {avg_color_str}")
                st.markdown(f"- **Shape**: {shape_comment}")
                st.markdown(f"- **Texture**: {texture_comment}")
                st.markdown("- **TCM Insight**: Qi Deficiency (based on image features)")
                st.markdown("- **Western Equivalent**: Signs of fatigue or low hemoglobin")

                feedback = st.text_input("How accurate was this? (optional feedback)")
                if feedback:
                    st.success("Thanks for your feedback! 🙏")
            else:
                st.error("⚠️ Please upload an image and provide consent.")

    elif page == "About & Disclaimer":
        st.title("📜 About This App")
        st.write("""
            This app is a prototype to:
            - Educate users on Traditional Chinese Medicine
            - Explore tongue analysis as a health indicator
            - Begin building a research dataset

            ⚠️ **Disclaimer**: This tool does NOT replace medical professionals. Use responsibly.
        """)
    else:
        st.warning("Page not recognized.")

except Exception as e:
    st.error("❌ Failed to load the page.")
    st.exception(e)
