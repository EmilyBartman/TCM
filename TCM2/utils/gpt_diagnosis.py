# utils/gpt_diagnosis.py
import openai
import base64
from PIL import Image
import io
import streamlit as st
from openai import OpenAI

def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def run_gpt_diagnosis(user_inputs, img_path=None):
    try:
        client = OpenAI(api_key=st.secrets["openai"]["api_key"])

        symptoms = ", ".join(user_inputs.get("symptoms", [])) or "None reported"
        tongue = user_inputs.get("tongue_characteristics", {})
        vitals = user_inputs.get("vitals", {})

        tongue_description = (
            f"Color: {tongue.get('color')}, "
            f"Shape: {tongue.get('shape')}, "
            f"Coating: {tongue.get('coating')}, "
            f"Moisture: {tongue.get('moisture')}, "
            f"Bumps: {tongue.get('bumps')}"
        )

        prompt = (
            f"A user has submitted a tongue image and health input for educational TCM-based insight.\n\n"
            f"---\n"
            f"Symptoms: {symptoms}\n"
            f"Tongue Description: {tongue_description}\n"
            f"Vitals: {vitals}\n"
            f"---\n"
            f"Please:\n"
            f"1. Suggest a likely TCM pattern (e.g., Qi Deficiency, Damp Heat)\n"
            f"2. Suggest a Western analogy (if applicable)\n"
            f"3. Suggest 2–3 lifestyle remedies\n"
            f"4. Note any discrepancies in the tongue image vs the self-report\n"
            f"5. Return a confidence score from 0–100\n"
            f"Respond ONLY in JSON: `tcm_syndrome`, `western_equivalent`, `remedies`, `discrepancies`, `confidence`"
        )

        messages = [
            {"role": "system", "content": "You are a multimodal TCM diagnostic assistant for wellness education."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(img_path)}"}}
            ]}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3
        )

        result = response.choices[0].message.content
        try:
            json_start = result.index("{")
            json_end = result.rindex("}") + 1
            return json.loads(result[json_start:json_end])
        except Exception:
            return result  # fallback

    except Exception as e:
        st.error("❌ GPT-4o image analysis failed.")
        st.exception(e)
        return None
