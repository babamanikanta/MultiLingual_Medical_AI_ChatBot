from deep_translator import GoogleTranslator
import re


# ----------------------------
# Language detection (IMPROVED + SAFE)
# ----------------------------
def detect_language_custom(text):
    if not text:
        return "en"

    text_lower = text.lower()

    # ================= TELUGU SCRIPT
    if re.search(r'[\u0C00-\u0C7F]', text):
        return "te"

    # ================= HINDI SCRIPT
    if re.search(r'[\u0900-\u097F]', text):
        return "hi"

    # ================= TENGGLISH DETECTION (FIX)
    telugu_words = [
        "valla", "ki", "undi", "le", "emi", "maa", "daggu", "jwaram"
    ]

    hindi_words = [
        "hai", "mera", "mujhe", "bukhar", "dard"
    ]

    t_score = sum(1 for w in telugu_words if w in text_lower)
    h_score = sum(1 for w in hindi_words if w in text_lower)

    if t_score >= 2:
        return "tenglish"

    if h_score >= 2:
        return "hinglish"

    return "en"


# ----------------------------
# Convert to English (SAFE + STABLE)
# ----------------------------
def to_english(text):
    lang = detect_language_custom(text)

    if lang == "en":
        return text, "en"

    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated, lang
    except:
        return text, lang

# ----------------------------
# Translate response back (SAFE OUTPUT)
# ----------------------------
def translate_to_user_lang(text, lang):

    if lang == "en":
        return text

    try:

        if lang == "te":
            return GoogleTranslator(source='auto', target='te').translate(text)

        if lang == "tenglish":
            return GoogleTranslator(source='auto', target='te').translate(text)

        if lang == "hi":
            return GoogleTranslator(source='auto', target='hi').translate(text)

        if lang == "hinglish":
            return GoogleTranslator(source='auto', target='hi').translate(text)

    except:
        return text