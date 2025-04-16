# utils/gpt_diagnosis.py
import base64
from PIL import Image
from openai import OpenAI
import json
import streamlit as st

def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def run_gpt_diagnosis(user_inputs, img_path):
    try:
        b64_img = image_to_base64(img_path)

        user_symptoms = ", ".join(user_inputs.get("symptoms", []))
        characteristics = user_inputs.get("tongue_characteristics", {})
        vitals = user_inputs.get("vitals", {})

        prompt = f"""
You are a Traditional Chinese Medicine (TCM) practitioner and diagnostic assistant. Given the following:
- A tongue image (attached below)
- User-reported symptoms: {user_symptoms}
- Tongue characteristics: {characteristics}
- Vitals: {vitals}

Instructions:
1. Analyze the image and suggest the most likely TCM syndrome.
2. Map that to a Western medical condition if applicable.
3. Suggest up to 3 lifestyle or dietary remedies based on TCM.
4. Compare the user-reported characteristics with what you observe in the image. Mention any discrepancies.
5. Provide a confidence score from 0 to 100 for the diagnosis.

Respond only in valid JSON with these keys: tcm_syndrome, western_equivalent, remedies, discrepancies, confidence.
"""

        client = OpenAI(api_key=st.secrets["openai"]["api_key"])

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a multimodal medical assistant that ALWAYS interprets the uploaded image before giving a diagnosis."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Here is the patient's tongue image. You MUST analyze it directly as part of the diagnosis. Please follow the steps described."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]}
            ]
            ,
            temperature=0.3
        )

        reply = response.choices[0].message.content
        try:
            start = reply.index("{")
            end = reply.rindex("}") + 1
            parsed = json.loads(reply[start:end])
            return parsed  # 💡 return as dict
        except Exception:
            return reply 

    except Exception as e:
        st.error("GPT diagnosis failed.")
        st.exception(e)
        return None

