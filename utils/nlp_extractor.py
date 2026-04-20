import spacy
import re
from utils.translator import to_english

nlp = spacy.load("en_core_web_sm")


SYMPTOM_EXPANSION = {
    "sir dard": "headache",
    "headache": "headache",
    "pet dard": "stomach_pain",
    "ulti": "vomiting",
    "bukhar": "high_fever",
    "fever": "high_fever",
    "khansi": "cough"
}


def phrase_exists(text, phrase):
    return phrase in text


def extract_symptoms_nlp(text, symptom_columns):

    if not text:
        return []

    # translate
    text, lang = to_english(text)
    text = re.sub(r'\s+', ' ', text.lower().strip())

    found = set()

    # ----------------------------
    # 1. EXPANSION (FIXED)
    # ----------------------------
    for key, value in SYMPTOM_EXPANSION.items():
        if phrase_exists(text, key):
            found.add(value)

    # ----------------------------
    # 2. DIRECT COLUMN MATCH
    # ----------------------------
    for col in symptom_columns:
        if col.replace("_", " ") in text:
            found.add(col)

    # ----------------------------
    # 3. NLP fallback
    # ----------------------------
    doc = nlp(text)

    for token in doc:
        if token.text in symptom_columns:
            found.add(token.text)

    return list(found)