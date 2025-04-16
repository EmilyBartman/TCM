# utils/gpt_diagnosis.py
import streamlit as st
from openai import OpenAI
import json

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
            f"A user has submitted self-reported tongue characteristics and symptoms.\n\n"
            f"Please help interpret the following from a Traditional Chinese Medicine (TCM) perspective, "
            f"strictly for educational and wellness insights (not diagnosis).\n\n"

            f"### Reported Tongue Description:\n{tongue_description}\n\n"
            f"### Reported Symptoms:\n{symptoms}\n\n"
            f"### Vitals / Lifestyle Info:\n{vitals}\n\n"

            f"1. Suggest the most likely TCM pattern (e.g., Qi Deficiency, Damp Heat).\n"
            f"2. Optionally map it to a general Western interpretation (e.g., fatigue, digestion issues).\n"
            f"3. Suggest 2–3 supportive remedies (e.g., warm foods, stress reduction).\n"
            f"4. If the tongue description is vague or inconsistent (e.g., pale but dry and swollen), flag that.\n"
            f"5. If this doesn't sound like a real tongue description, politely ask for a clearer tongue photo upload.\n"
            f"6. Estimate confidence (0–100) in your recommendation.\n\n"
            f"Respond ONLY in valid JSON with these keys: "
            f"`tcm_syndrome`, `western_equivalent`, `remedies`, `discrepancies`, `confidence`."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a friendly, wellness-focused Traditional Chinese Medicine assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content

        # Attempt to parse structured JSON output
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            return json.loads(content[start:end])
        except Exception:
            return content  # fallback to raw output

    except Exception as e:
        st.error("GPT diagnosis failed.")
        st.exception(e)
        return None
