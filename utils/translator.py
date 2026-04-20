from deep_translator import GoogleTranslator
import re


# ----------------------------
# Language detection (IMPROVED + SAFE)
# ----------------------------
def detect_language_custom(text):
    if not text:
        return "en"

    text = text.lower().strip()

    # Telugu script
    if re.search(r'[\u0C00-\u0C7F]', text):
        return "te"

    # Hindi script
    if re.search(r'[\u0900-\u097F]', text):
        return "hi"

    # Telugu roman keywords (expanded slightly)
    telugu_words = [
        "jwaram", "noppi", "talanoppi", "naku", "undi",
        "durada", "mukku", "daggu", "vanta"
    ]

    # Hindi roman keywords (expanded slightly)
    hindi_words = [
        "bukhar", "khansi", "dard", "hai", "pet", "sir"
    ]

    words = text.split()

    if any(word in text for word in telugu_words):
        return "te"

    if any(word in text for word in hindi_words):
        return "hi"

    return "en"


# ----------------------------
# Convert to English (SAFE + STABLE)
# ----------------------------
def to_english(text):
    lang = detect_language_custom(text)

    if not text:
        return "", "en"

    # already English
    if lang == "en":
        return text, "en"

    try:
        translated = GoogleTranslator(source=lang, target='en').translate(text)

        # safety checks (VERY IMPORTANT)
        if (
            not translated
            or len(translated.strip()) < 2
            or translated.strip().lower() == text.strip().lower()
        ):
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