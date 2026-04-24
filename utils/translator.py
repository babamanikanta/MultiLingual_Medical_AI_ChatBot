from deep_translator import GoogleTranslator
import re


# ----------------------------
# Language detection (IMPROVED + SAFE)
# ----------------------------
def detect_language_custom(text):
    if not text:
        return "en"

    text_lower = text.lower()

    # 1. Telugu script (strong signal)
    if re.search(r'[\u0C00-\u0C7F]', text_lower):
        return "te"

    # 2. Hindi script (strong signal)
    if re.search(r'[\u0900-\u097F]', text_lower):
        return "hi"

    # 3. Roman Telugu signals (VERY SIMPLE)
    telugu_signals = ["maa", "naaku", "undi", "tala", "noppi", "vundi", "vundi"]

    # 4. Roman Hindi signals (VERY GENERIC — no language knowledge needed)
    hindi_signals = ["hai", "aur", "ki", "me", "se"]

    # scoring
    telugu_score = sum(1 for w in telugu_signals if w in text_lower)
    hindi_score = sum(1 for w in hindi_signals if w in text_lower)

    # IMPORTANT RULE:
    # If Telugu-style words exist → prefer Telugu
    if telugu_score > 0:
        return "te"

    # else if Hindi-style structure → Hindi
    if hindi_score > 0:
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