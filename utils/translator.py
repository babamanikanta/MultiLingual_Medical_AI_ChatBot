from deep_translator import GoogleTranslator
import re


# ----------------------------
# Language detection (IMPROVED + SAFE)
# ----------------------------
def detect_language_custom(text):
    if not text:
        return "en"

    text_lower = text.lower()

    # ----------------------------
    # 1. Strong script detection
    # ----------------------------
    if re.search(r'[\u0C00-\u0C7F]', text_lower):
        return "te"

    if re.search(r'[\u0900-\u097F]', text_lower):
        return "hi"

    # ----------------------------
    # 2. Strong Hinglish detection
    # ----------------------------
    hindi_keywords = [
        "hai", "hun", "ho", "tha", "thi", "the",
        "mera", "meri", "mere",
        "dost", "dosth",
        "mujhe", "mujko", "mujhko",
        "ko", "aur",
        "bukhar", "khansi", "ulti", "dard", "pet"
    ]

    # count matches
    score = sum(1 for word in hindi_keywords if word in text_lower)

    # 🔥 KEY RULE: if meaningful Hindi words exist → Hindi
    if score >= 2:
        return "hi"

    return "en"


# ----------------------------
# Convert to English (SAFE + STABLE)
# ----------------------------
def to_english(text):
    lang = detect_language_custom(text)

    if not text:
        return "", "en"

    if lang == "en":
        return text, "en"

    try:
        translated = GoogleTranslator(source=lang, target='en').translate(text)

        # 🔥 BETTER SAFETY CHECK
        if not translated or translated.strip().lower() == text.strip().lower():
            return text, lang

        return translated, lang

    except Exception as e:
        print("Translation error:", e)
        return text, lang


# ----------------------------
# Translate response back (SAFE OUTPUT)
# ----------------------------
def translate_to_user_lang(text, target_lang):

    if not text:
        return ""

    if target_lang == "en":
        return text

    try:
        translated = GoogleTranslator(
            source='en',
            target=target_lang
        ).translate(text)

        # safety check
        if not translated or len(translated.strip()) < 2:
            return text

        return translated

    except Exception as e:
        print("Reverse translation error:", e)
        return text