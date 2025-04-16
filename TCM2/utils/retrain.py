# utils/retrain.py

import os
import joblib
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from sklearn.linear_model import LogisticRegression
from utils.model_utils import extract_features

def retrain_model_from_feedback(db):
    print("🔄 Retraining model from feedback...")

    # Fetch feedback docs that have syndrome corrections
    feedback_docs = db.collection("medical_feedback").stream()

    X = []
    y = []

    for doc in feedback_docs:
        feedback = doc.to_dict()
        corrections = feedback.get("corrections", {})
        syndrome = corrections.get("tcm_syndrome", "").strip()
        if not syndrome:
            continue  # Skip if no corrected label

        submission_id = feedback.get("submission_id")
        if not submission_id:
            continue

        user_doc = db.collection("tongue_submissions").document(submission_id).get().to_dict()
        if not user_doc or "image_url" not in user_doc:
            continue

        try:
            img_url = user_doc["image_url"]
            response = requests.get(img_url)
            if response.status_code != 200:
                continue

            img = Image.open(BytesIO(response.content))
            with BytesIO() as tmp_io:
                img.save(tmp_io, format="JPEG")
                tmp_io.seek(0)
                # Save to disk temporarily for extract_features
                with open("tmp_train.jpg", "wb") as f:
                    f.write(tmp_io.read())

            features = extract_features("tmp_train.jpg")
            X.append(features)
            y.append(syndrome)

        except Exception as e:
            print(f"❌ Skipped {submission_id}: {e}")
            continue

    if not X or not y:
    print("⚠️ No valid data found for retraining.")
    return

    if len(set(y)) < 2:
        print("❌ Cannot train: need at least two distinct classes. Found:", set(y))
        return
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/tcm_diagnosis_model.pkl")
    print("✅ Model retrained and saved to models/tcm_diagnosis_model.pkl")
