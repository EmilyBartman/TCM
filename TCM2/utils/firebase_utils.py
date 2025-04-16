# utils/firebase_utils.py
import os
import cv2
import tempfile
from datetime import timedelta
import firebase_admin  
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

# Upload file to Firebase and return URL
def upload_image_to_firebase(uploaded_file, submission_id, bucket):
    import tempfile
    import uuid

    # Save image to a temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    temp_file.flush()

    # Define path in Firebase Storage
    blob_path = f"tongue_images/{submission_id}.jpg"
    blob = bucket.blob(blob_path)

    # Upload file
    blob.upload_from_filename(temp_file.name)

    # Generate permanent token-based public URL
    token = str(uuid.uuid4())
    blob.metadata = {"firebaseStorageDownloadTokens": token}
    blob.patch()

    public_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{blob_path.replace('/', '%2F')}?alt=media&token={token}"

    return public_url, temp_file.name


# Save user inputs to Firestore
def save_user_submission(submission_id, timestamp, image_url, user_inputs, db):
    db.collection("tongue_submissions").document(submission_id).set({
        "submission_id": submission_id,
        "timestamp": timestamp,
        "image_url": image_url,
        "user_inputs": user_inputs
    })
