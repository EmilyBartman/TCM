# utils/gpt_diagnosis.py
import base64
import json
from openai import OpenAI
import streamlit as st

def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def run_gpt_diagnosis(user_inputs, img_path):
    try:
        b64_img = image_to_base64(img_path)

        client = OpenAI(api_key=st.secrets["openai"]["api_key"])

        prompt = (
            "Please analyze the uploaded **image of a human tongue** from a Traditional Chinese Medicine perspective.\n"
            "You MUST analyze the image — do not skip it. Use this image to:\n"
            "1. Identify a TCM syndrome.\n"
            "2. Suggest a corresponding Western interpretation (if any).\n"
            "3. Recommend up to 3 remedies.\n"
            "4. Compare the tongue image with user-reported details.\n"
            "5. Give a confidence score (0-100).\n\n"
            f"User-reported symptoms: {', '.join(user_inputs.get('symptoms', []))}\n"
            f"Tongue characteristics: {user_inputs.get('tongue_characteristics', {})}\n"
            f"Vitals: {user_inputs.get('vitals', {})}\n\n"
            "Respond only with a JSON object using these keys:\n"
            "`tcm_syndrome`, `western_equivalent`, `remedies`, `discrepancies`, `confidence`."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical diagnostic assistant trained in both Traditional Chinese Medicine and Western medicine."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_img}"
                    }}
                ]}
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content

        # Try parsing JSON from response
        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        parsed = json.loads(json_str)
        return parsed

    except Exception as e:
        st.warning("⚠️ GPT-4o did not return structured output.")
        return content if "content" in locals() else str(e)
