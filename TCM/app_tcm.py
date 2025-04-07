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
            submission_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            file_ext = uploaded_img.name.split(".")[-1]
            firebase_filename = f"tongue_images/{submission_id}.{file_ext}"

            os.makedirs("temp", exist_ok=True)
            temp_path = f"temp/{submission_id}.{file_ext}"
            img.save(temp_path)

            cv_img = cv2.imread(temp_path)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(cv_img, (300, 300))

            avg_color = np.mean(resized.reshape(-1, 3), axis=0)
            avg_color_str = f"RGB({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})"

            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            shape_comment = "Swollen or Elongated" if edge_pixels > 5000 else "Normal"
            texture_comment = "Dry/Coated" if laplacian_var > 100 else "Moist"

            prediction_TCM = ""
            prediction_Western = ""

            if avg_color[0] < 180 and "dry" in texture_comment.lower():
                prediction_TCM = "Yin Deficiency"
                prediction_Western = "Possible dehydration or hormone imbalance"
            elif "Swollen" in shape_comment:
                prediction_TCM = "Damp Retention"
                prediction_Western = "Inflammation or fluid retention"
            elif avg_color[0] < 140 and avg_color[2] > 160:
                prediction_TCM = "Blood Deficiency"
                prediction_Western = "Anemia or nutritional deficiency"
            else:
                prediction_TCM = "Qi Deficiency"
                prediction_Western = "Low energy, fatigue, low immunity"

            try:
                blob = bucket.blob(firebase_filename)
                blob.upload_from_filename(temp_path)
                url = blob.generate_signed_url(expiration=timedelta(hours=1), method="GET")
                img_url = url
                st.success("✅ Image uploaded.")
                st.write("🔗 Temporary image URL:", img_url)
            except Exception as e:
                st.error("❌ Upload to Firebase failed.")
                st.exception(e)
                st.stop()

            result = {
                "id": submission_id,
                "timestamp": timestamp,
                "symptoms": symptoms,
                "tongue_image_url": img_url,
                "avg_color": avg_color_str,
                "shape_comment": shape_comment,
                "texture_comment": texture_comment,
                "prediction_TCM": prediction_TCM,
                "prediction_Western": prediction_Western,
                "user_feedback": ""
            }
            st.session_state.submissions.append(result)
            db.collection("tongue_scans").document(submission_id).set(result)

            st.subheader("🧪 Analysis Results")

            with st.container():
                st.markdown(f"**🖌️ Tongue Color**
:white_circle: {avg_color_str}")
                st.markdown("""
                - This is a soft reddish tone, commonly seen in people who may feel tired or low-energy. 
                - In TCM, it suggests Qi or Blood Deficiency.
                """)

                st.markdown(f"**📐 Shape**
:straight_ruler: {shape_comment}")
                st.markdown("""
                - A 'Normal' shape means the tongue edges are smooth and not swollen or too thin. 
                - This usually means there's no extreme heat or cold imbalance.
                """)

                st.markdown(f"**🌫️ Texture**
:droplet: {texture_comment}")
                st.markdown("""
                - A moist texture suggests your body is doing a good job of keeping fluids balanced. 
                - Too dry or too wet may hint at Yin issues or Dampness.
                """)

                st.markdown(f"**🧧 TCM Insight**
:red_gift_envelope: {prediction_TCM}")
                st.markdown("""
                - This pattern suggests your body's energy (Qi) might be a bit low. 
                - You may feel tired, have cold limbs, or weak digestion.
                - TCM may suggest warm foods, rest, or herbal tea.
                """)

                st.markdown(f"**🧬 Western Insight**
:microscope: {prediction_Western}")
                st.markdown("""
                - These signs might relate to low iron, fatigue, or stress-related tiredness. 
                - Improving hydration, nutrition, or rest often helps.
                """)

            feedback = st.radio("Was this prediction accurate?", ["Yes", "No", "Not sure"], index=2)
            if feedback != "Not sure":
                db.collection("tongue_scans").document(submission_id).update({
                    "user_feedback": feedback,
                    "is_correct": True if feedback == "Yes" else False
                })
                st.toast("🙏 Thanks for your feedback!", icon="💬")"Was this prediction accurate?", ["Yes", "No", "Not sure"], index=2)
            if feedback:
                db.collection("tongue_scans").document(submission_id).update({
                    "user_feedback": feedback,
                    "is_correct": True if feedback == "Yes" else (False if feedback == "No" else None)
                })
                st.success("🙏 Thanks for your feedback!")

# ---- SUBMISSION HISTORY ----
elif page == "Submission History":
    st.title("📜 My Tongue Scan History")
    if st.session_state.submissions:
        df = pd.DataFrame(st.session_state.submissions)

        st.dataframe(df)

        # Accuracy summary if feedback exists
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
