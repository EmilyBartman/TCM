# utils/translation.py
import streamlit as st
from deep_translator import GoogleTranslator

LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "French": "fr",
    "Hindi": "hi",
    "Arabic": "ar",
    "Swahili": "sw",
    "Zulu": "zu",
    "Amharic": "am",
    "Igbo": "ig",
    "Yoruba": "yo",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
    "Bengali": "bn",
    "Malay": "ms",
    "Vietnamese": "vi",
    "Thai": "th",
    "Filipino": "fil",
    "Japanese": "ja",
    "Korean": "ko"
}

def set_language_selector():
    if "selected_lang" not in st.session_state:
        st.session_state.selected_lang = "English"

    new_lang = st.sidebar.selectbox("🌐 Choose Language", list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.selected_lang))
    if new_lang != st.session_state.selected_lang:
        st.session_state.selected_lang = new_lang
        st.rerun()
    return LANGUAGES[st.session_state.selected_lang]

def translate(text, lang_code):
    try:
        return GoogleTranslator(source="auto", target=lang_code).translate(text)
    except Exception:
        return text
