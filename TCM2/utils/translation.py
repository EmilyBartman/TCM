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
    # Default to "English" on first load
    if "selected_lang" not in st.session_state or st.session_state.selected_lang not in LANGUAGES:
        st.session_state.selected_lang = "English"

    # Translate label using fallback if available
    label_lang_selector = "🌐 Choose Language"
    try:
        fallback_code = LANGUAGES.get(st.session_state.selected_lang, "en")
        label_lang_selector = translate("🌐 Choose Language", fallback_code)
    except:
        pass

    # Let user select language by native name
    new_lang = st.sidebar.selectbox(
        label_lang_selector,
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.selected_lang)
    )

    if new_lang != st.session_state.selected_lang:
        st.session_state.selected_lang = new_lang
        st.rerun()

    return LANGUAGES[st.session_state.selected_lang]


def translate(text, lang_code):
    try:
        return GoogleTranslator(source="auto", target=lang_code).translate(text)
    except Exception:
        return text
