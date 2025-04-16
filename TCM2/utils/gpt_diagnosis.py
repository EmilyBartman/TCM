# utils/gpt_diagnosis.py
import base64
import json
from openai import OpenAI
import streamlit as st

def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def run_gpt_diagnosis(user_inputs, img_path):
    from openai import OpenAI
    import json, base64, io
    import streamlit as st

    try:
        with open(img_path, "rb") as image_file:
            b64_img = base64.b64encode(image_file.read()).decode("utf-8")

        client = OpenAI(api_key=st.secrets["openai"]["api_key"])

        user_symptoms = ", ".join(user_inputs.get("symptoms", []))
        characteristics = user_inputs.get("tongue_characteristics", {})
        vitals = user_inputs.get("vitals", {})

        prompt = (
            "You are a Traditional Chinese Medicine (TCM) assistant helping users better understand general wellness patterns. "
            "You are not a doctor, and you do not diagnose medical conditions. This is for educational purposes only.\n\n"
            "Please analyze the tongue image and user inputs below. Based on the visual features and reported characteristics:\n"
            "1. Identify a **TCM pattern** (e.g., Qi Deficiency, Damp Heat)\n"
            "2. Suggest a **general wellness analogy** (e.g., fatigue, digestion imbalance)\n"
            "3. Recommend **lifestyle or food habits** that might support balance (e.g., warm soups, rest, less greasy food)\n"
            "4. Note any differences between image signs and user reports\n"
            "5. Provide a confidence estimate out of 100 for how well the signs align\n\n"
            "Respond in JSON only using these keys: `tcm_syndrome`, `western_equivalent`, `remedies`, `discrepancies`, `confidence`."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a wellness coach trained in Traditional Chinese Medicine."},
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

        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            return json.loads(content[start:end])
        except Exception:
            return content

    except Exception as e:
        st.error("GPT diagnosis failed.")
        st.exception(e)
        return None

