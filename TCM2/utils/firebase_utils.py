# utils/firebase_utils.py
import os
import cv2
import tempfile
from datetime import timedelta
from tempfile import NamedTemporaryFile
import os
import firebase_admin  
from datetime import timedelta
from firebase_admin import credentials, firestore, storage, initialize_app
import streamlit as st

# Initialize Firebase
@st.cache_resource
def init_firebase():
    try:
        firebase_config = dict(st.secrets["firebase"])
        cred = credentials.Certificate(firebase_config)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                "storageBucket": "traditional-medicine-50518"  # ← hardcoded like old version
            })

        db = firestore.client()
        bucket = storage.bucket("traditional-medicine-50518")  # ← match exactly
        return db, bucket

    except Exception as e:
        st.error("Firebase init failed")
        st.exception(e)
        return None, None



def upload_image_to_firebase(uploaded_img, submission_id, bucket):
    uploaded_img.seek(0)
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_img.read())
        tmp_path = tmp.name

    if os.path.getsize(tmp_path) == 0:
        raise ValueError("Uploaded image is empty. Please upload a valid image.")

    blob_path = f"tongue_images/{submission_id}.jpg"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(tmp_path)

    # 🔓 Make image publicly readable
    blob.make_public()

    # 🔥 Return the permanent public URL
    return blob.public_url, tmp_path



# Save user inputs to Firestore
def save_user_submission(submission_id, timestamp, image_url, user_inputs, db):
    db.collection("tongue_submissions").document(submission_id).set({
        "submission_id": submission_id,
        "timestamp": timestamp,
        "image_url": image_url,
        "user_inputs": user_inputs
    })
