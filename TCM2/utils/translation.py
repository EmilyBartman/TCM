# utils/translation.py
import streamlit as st
import re
from deep_translator import GoogleTranslator

LANGUAGES = {
    "English": "en",
    "Español": "es",
    "简体中文": "zh-CN",          # Chinese (Simplified)
    "繁體中文": "zh-TW",          # Chinese (Traditional)
    "Français": "fr",
    "हिन्दी": "hi",              # Hindi
    "العربية": "ar",            # Arabic
    "Kiswahili": "sw",          # Swahili
    "IsiZulu": "zu",            # Zulu
    "አማርኛ": "am",              # Amharic
    "Igbo": "ig",
    "Yorùbá": "yo",
    "தமிழ்": "ta",               # Tamil
    "తెలుగు": "te",             # Telugu
    "اردو": "ur",               # Urdu
    "বাংলা": "bn",              # Bengali
    "Bahasa Melayu": "ms",      # Malay
    "Tiếng Việt": "vi",         # Vietnamese
    "ไทย": "th",                # Thai
    "Filipino": "fil",
    "日本語": "ja",              # Japanese
    "한국어": "ko"               # Korean
}


    
def set_language_selector():
    if "selected_lang" not in st.session_state:
        st.session_state.selected_lang = "English"

    lang_names = list(LANGUAGES.keys())
    current_lang = st.session_state.selected_lang

    new_lang = st.sidebar.selectbox(
        "🌐 Choose Language",
        lang_names,
        index=lang_names.index(current_lang)
    )

    if new_lang != current_lang:
        st.session_state.selected_lang = new_lang
        # prevent rerun from resetting page selection
        st.rerun()
    return LANGUAGES[st.session_state.selected_lang]


def clean_markdown(text):
    # Strip markdown elements that confuse the translator
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"#+ ", "", text)  # headers like ### or ##
    return text


def translate(text, lang_code):
    try:
        clean_text = clean_markdown(text)
        translated = GoogleTranslator(source="auto", target=lang_code).translate(clean_text)
        return translated
    except Exception:
        return text

