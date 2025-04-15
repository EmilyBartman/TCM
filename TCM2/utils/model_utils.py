# utils/model_utils.py
import os
import torch
import joblib
import numpy as np
from torchvision import models, transforms
from sklearn.linear_model import LogisticRegression

# Load feature extractor (MobileNet)
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
def extract_features(image_path):
    import cv2
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0)
    features = mobilenet(img_tensor).squeeze().numpy()
    return features.tolist()

def load_model():
    model_path = "models/tcm_diagnosis_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

def predict_with_model(model, features):
    try:
        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0].max()
        return pred, round(prob * 100, 2)
    except Exception:
        return "Model Error", 0

def compare_user_vs_model(user_inputs, model_outputs):
    differences = {}
    confidence_hits = 0
    total_fields = 0

    for section in ["symptoms", "tongue_characteristics"]:
        user_vals = user_inputs.get(section, {})
        model_vals = model_outputs.get(section, {})

        if isinstance(user_vals, list):
            matched = set(user_vals).intersection(set(model_vals))
            missed = set(user_vals) - set(model_vals)
            extra = set(model_vals) - set(user_vals)

            differences[section] = {
                "matched": list(matched),
                "missed_by_model": list(missed),
                "extra_predicted": list(extra)
            }
            confidence_hits += len(matched)
            total_fields += len(set(user_vals).union(set(model_vals)))
        else:
            section_diff = {}
            for key in user_vals:
                u_val = str(user_vals[key]).strip().lower()
                m_val = str(model_vals.get(key, "")).strip().lower()
                section_diff[key] = {
                    "user": u_val,
                    "model": m_val,
                    "match": u_val == m_val
                }
                if u_val == m_val:
                    confidence_hits += 1
                total_fields += 1
            differences[section] = section_diff

    score = round((confidence_hits / max(total_fields, 1)) * 100, 2)
    return differences, score

def get_remedies(tcm_pattern):
    remedies_map = {
        "Qi Deficiency": ["Ginseng tea", "Sweet potatoes", "Walking"],
        "Yin Deficiency": ["Goji berries", "Pears and lily bulb soup", "Meditation"],
        "Blood Deficiency": ["Beets", "Spinach", "Dang Gui"],
        "Damp Retention": ["Barley water", "Avoid greasy food", "Pu-erh tea"]
    }
    return remedies_map.get(tcm_pattern, ["Hydration", "Rest", "Balanced diet"])
