# utils/translation.py
import streamlit as st
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

    label_lang = "🌐 Choose Language"
    if "selected_lang" in st.session_state:
        try:
            fallback_lang_code = LANGUAGES[st.session_state.selected_lang]
            label_lang = translate("🌐 Choose Language", fallback_lang_code)
        except:
            pass
    
    new_lang = st.sidebar.selectbox(label_lang, list(LANGUAGES.keys()),
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
