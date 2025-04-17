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
    print("🔄 Collecting expert corrections...")

    feedback_docs = db.collection("medical_feedback").stream()

    count = 0
    for doc in feedback_docs:
        feedback = doc.to_dict()
        if feedback.get("corrections", {}):
            count += 1

    print(f"✅ {count} expert feedback submissions collected.")
