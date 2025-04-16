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
            "You are a wellness AI assistant trained in Traditional Chinese Medicine. "
            "You are helping someone better understand their wellness from a non-diagnostic, educational perspective.\n\n"
            "Use the image of their tongue below, as well as their described symptoms and characteristics, to:\n"
            "1. Identify a likely TCM pattern (e.g., Qi Deficiency, Damp Retention, etc.)\n"
            "2. Offer a possible Western analogy, like fatigue or digestive imbalance\n"
            "3. Suggest 2–3 general remedies (like rest, food, herbal tea)\n"
            "4. Compare what you observe in the image vs. what the user reported\n"
            "5. Estimate a confidence score out of 100\n\n"
            "Respond using valid JSON with these keys only:\n"
            "`tcm_syndrome`, `western_equivalent`, `remedies`, `discrepancies`, `confidence`."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a friendly, educational health assistant for TCM wellness."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_img}"
                    }}
                ]}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content

        # Attempt to extract JSON
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            return json.loads(content[start:end])
        except Exception:
            return content  # Fallback to raw text

    except Exception as e:
        st.error("GPT diagnosis failed.")
        st.exception(e)
        return None

    except Exception as e:
        st.warning("⚠️ GPT-4o did not return structured output.")
        return content if "content" in locals() else str(e)
