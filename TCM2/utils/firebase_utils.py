# utils/firebase_utils.py
import os
import cv2
import tempfile
from datetime import timedelta
from firebase_admin import credentials, firestore, storage, initialize_app
import streamlit as st

# Initialize Firebase
@st.cache_resource
def init_firebase():
    try:
        firebase_config = dict(st.secrets["firebase"])
        cred = credentials.Certificate(firebase_config)
        if not len(initialize_app._apps):
            initialize_app(cred, {
                "storageBucket": firebase_config["project_id"] + ".appspot.com"
            })
        return firestore.client(), storage.bucket()
    except Exception as e:
        st.error("Firebase init failed")
        st.exception(e)
        return None, None

# Upload file to Firebase and return URL
def upload_image_to_firebase(uploaded_img, submission_id, bucket):
    extension = uploaded_img.name.split(".")[-1]
    file_name = f"{submission_id}.{extension}"
    firebase_path = f"tongue_images/{file_name}"

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file_name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_img.getbuffer())

    blob = bucket.blob(firebase_path)
    blob.upload_from_filename(temp_path)
    url = blob.generate_signed_url(expiration=timedelta(hours=1), method="GET")
    return url, temp_path

# Save user inputs to Firestore
def save_user_submission(submission_id, timestamp, image_url, user_inputs, db):
    db.collection("tongue_submissions").document(submission_id).set({
        "submission_id": submission_id,
        "timestamp": timestamp,
        "image_url": image_url,
        "user_inputs": user_inputs
    })
