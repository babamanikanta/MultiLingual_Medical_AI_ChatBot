import pandas as pd
import joblib
import re
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

from utils.data_loader import get_disease_info
from utils.translator import translate_to_user_lang, to_english
from utils.nlp_extractor import extract_symptoms_nlp

# ----------------------------
# Load model + dataset
# ----------------------------
model = joblib.load("model/disease_model.pkl")

train_df = pd.read_csv("data/training_improved.csv")
train_df.columns = train_df.columns.str.strip().str.replace(" ", "_")

symptom_columns = train_df.columns[:-1].tolist()
ALL_DISEASES = set(model.classes_)
ALL_DISEASES_LOWER = {d.lower() for d in ALL_DISEASES}

# ----------------------------
# Configure Gemini API
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

model_gemini = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model_gemini = genai.GenerativeModel("gemini-2.5-flash")
        print("✅ Gemini API is working")
    except Exception as e:
        print(f"⚠️ Gemini init error: {e}")
        model_gemini = None


# ----------------------------
# Disease Control
# ----------------------------
COMMON_DISEASES = {
    "Common Cold", "Viral Fever", "Flu", "Allergy", "Migraine", "Gastroenteritis"
}

SERIOUS_DISEASES = {
    "AIDS", "Paralysis (brain hemorrhage)", "Tuberculosis", "Cancer", "Heart attack"
}

# ----------------------------
# SYMPTOM MAP
# ----------------------------
RAW_SYMPTOM_MAP = {

    # ===================== FEVER =====================
    # English
    "fever": "high_fever",
    "high fever": "high_fever",

    # Hindi
    "bukhar": "high_fever",
    "jwaram": "high_fever",
    "jwar": "high_fever",
    "tez bukhar": "high_fever",

    # Telugu
    "జ్వరం": "high_fever",
    "jwaram": "high_fever",

    # ===================== COUGH =====================
    # English
    "cough": "cough",

    # Hindi
    "khansi": "cough",
    "khasi": "cough",

    # Telugu
    "దగ్గు": "cough",
    "daggu": "cough",

    # ===================== COLD / SNEEZING =====================
    # English
    "cold": "continuous_sneezing",

    # Hindi
    "sardi": "continuous_sneezing",
    "jalubu": "continuous_sneezing",
    "thand lagna": "chills",

    # Telugu
    "జలుబు": "continuous_sneezing",
    "cheemidi": "runny_nose",
    "ముక్కు కారుతుంది": "runny_nose",

    # ===================== RUNNY NOSE =====================
    "runny nose": "runny_nose",
    "mukku karutundi": "runny_nose",
    "mukku nundi neeru vastundi": "runny_nose",

    # ===================== HEADACHE =====================
    # English
    "headache": "headache",
    "head ache": "headache",

    # Hindi
    "sar dard": "headache",
    "sir dard": "headache",

    # Telugu
    "తలనొప్పి": "headache",
    "tala noppi": "headache",

    # ===================== STOMACH PAIN =====================
    # English
    "stomach pain": "stomach_pain",
    "abdominal pain": "abdominal_pain",

    # Hindi
    "pet dard": "stomach_pain",
    "pet mein dard": "stomach_pain",
    "pet me dard":"stomach_pain",

    # Telugu
    "కడుపు నొప్పి": "stomach_pain",
    "kadupu noppi": "stomach_pain",

    # ===================== VOMITING =====================
    # English
    "vomiting": "vomiting",
    "vomit": "vomiting",

    # Hindi
    "ulti": "vomiting",
    "ulti ho rahi": "vomiting",

    # Telugu
    "వాంతులు": "vomiting",
    "vanti": "vomiting",

    # ===================== DIARRHOEA =====================
    # English
    "diarrhea": "diarrhoea",
    "loose motions": "diarrhoea",

    # Hindi
    "dast": "diarrhoea",
    "loose motion": "diarrhoea",

    # Telugu
    "డయేరియా": "diarrhoea",
    "dayeriya": "diarrhoea",

    # ===================== WEAKNESS =====================
    # English
    "fatigue": "fatigue",
    "weakness": "fatigue",

    # Hindi
    "kamzori": "fatigue",
    "thakan": "fatigue",
    "bahut kamzori": "fatigue",

    # Telugu
    "బలహీనత": "fatigue",
    "weakness": "fatigue",

    # ===================== DIZZINESS =====================
    # English
    "dizziness": "dizziness",

    # Hindi
    "chakkar": "dizziness",
    "chakkar aana": "dizziness",

    # Telugu
    "తల తిరగడం": "dizziness",
    "chakkar": "dizziness",

    # ===================== CHEST PAIN =====================
    "chest pain": "chest_pain",
    "chest pain": "chest_pain",

    # ===================== NAUSEA =====================
    "nausea": "nausea",
    "pet me dard hai": "stomach_pain",
"pet mein dard hai": "stomach_pain",
"pet dard hai": "stomach_pain",
"aur": None,
"ulti ho rahi hai": "vomiting",
"ulti aa rahi hai": "vomiting",
"pet me dard ho raha hai": "stomach_pain",
"pet dard ho raha hai": "stomach_pain",
"mere pet me dard hai": "stomach_pain",
"mujhe ulti ho rahi hai": "vomiting",
}

SYMPTOM_MAP = {k: v for k, v in RAW_SYMPTOM_MAP.items() if v in symptom_columns}


# ----------------------------
# NORMALIZATION
# ----------------------------
def normalize_input(text):
    text = text.lower()

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    symptoms = set()

    # ✅ 🔥 CRITICAL FIX: SPLIT FIRST
    parts = re.split(r"\b(?:aur|and|,)\b", text)

    for part in parts:
        part = part.strip()

        # STEP 1: direct match (ignore spaces)
        clean_part = part.replace(" ", "")
        for key, value in SYMPTOM_MAP.items():
            if value and key.replace(" ", "") in clean_part:
                symptoms.add(value)

        # STEP 2: normal match
        for key, value in SYMPTOM_MAP.items():
            if value and key in part:
                symptoms.add(value)

        # STEP 3: fallback (VERY IMPORTANT)
        if "pet" in part and "dard" in part:
            symptoms.add("stomach_pain")

    # STEP 4: dataset column match (whole text)
    for col in symptom_columns:
        if col.replace("_", " ") in text:
            symptoms.add(col)

    # STEP 5: NLP (optional)
    try:
        symptoms.update(extract_symptoms_nlp(text, symptom_columns))
    except:
        pass

    return list(symptoms)


# ----------------------------
# RULE ENGINE
# ----------------------------
def apply_medical_rules(symptoms):
    s = set(symptoms)

    if "runny_nose" in s or "continuous_sneezing" in s:
        return ["Common Cold", "Allergy"]

    if "high_fever" in s and "cough" in s:
        return ["Viral Fever", "Flu"]

    if "headache" in s and "vomiting" in s:
        return ["Migraine"]

    # 🔥 NEW CRITICAL RULE
    if "stomach_pain" in s and "vomiting" in s:
        return ["Gastroenteritis", "Food Poisoning"]

    if ("stomach_pain" in s or "abdominal_pain" in s) and "diarrhoea" in s:
        return ["Gastroenteritis"]

    if "chest_pain" in s and "breathlessness" in s:
        return ["Heart attack"]

    return []

# ----------------------------
# ML PREDICTION
# ----------------------------
def predict(symptoms_list):
    input_data = pd.DataFrame([[0] * len(symptom_columns)], columns=symptom_columns)

    for s in symptoms_list:
        if s in symptom_columns:
            input_data.loc[0, s] = 1

    probs = model.predict_proba(input_data)[0]
    classes = model.classes_

    results = []

    for i, disease in enumerate(classes):
        confidence = probs[i] * 100

        if disease == "Heart attack":
            if not any(x in symptoms_list for x in ["chest_pain", "breathlessness"]):
                continue

        if disease in SERIOUS_DISEASES and confidence < 65:
            continue

        if disease in COMMON_DISEASES:
            confidence += 10

        if confidence >= 20:
            results.append({
                "disease": disease,
                "confidence": round(min(confidence, 100), 2)
            })

    return sorted(results, key=lambda x: x["confidence"], reverse=True)[:3]


# ----------------------------
# DOCTOR ADVICE
# ----------------------------
def get_doctor_advice(symptoms, predictions):
    s = set(symptoms)

    if "chest_pain" in s or "breathlessness" in s:
        return "🚨 Seek immediate medical attention."

    if "high_fever" in s:
        return "⚠️ If fever lasts more than 2-3 days, consult a doctor."

    if "vomiting" in s and "diarrhoea" in s:
        return "⚠️ Risk of dehydration."

    if predictions and predictions[0]["confidence"] < 60:
        return "⚠️ Diagnosis uncertain."

    return "✅ Monitor symptoms."


# ----------------------------
# FINAL RESPONSE BUILDER
# ----------------------------
def build_original_response(predictions, doctor_advice, lang, gemini_text=None):
    response = ""

    if gemini_text:
        response += f"""💬 AI Response:
{gemini_text}

----------------------------------------

"""

    for p in predictions:
        info = get_disease_info(p["disease"])

        response += f"""🦠 Disease: {p['disease']}
📊 Confidence: {p['confidence']}%

🧾 Description:
{info['description']}

💊 Medication:
{info['medication']}

🥗 Diet:
{info['diet']}

🛡️ Precautions:
{', '.join(info['precautions'])}

🏃 Workout:
{', '.join(info['workout'])}

----------------------------------------

"""

    response += f"""
🩺 Doctor Recommendation:
{doctor_advice}
"""

    return translate_to_user_lang(response, lang)
# ----------------------------
# SYMPTOM EXTRACTOR (FOR TELEGRAM + DB)
# ----------------------------
def extract_all_symptoms(user_input):
    # Step 1: original text
    symptoms = normalize_input(user_input)

    # Step 2: translated text
    translated_text, _ = to_english(user_input)

    if translated_text:
        symptoms = list(set(symptoms) | set(normalize_input(translated_text)))

    return symptoms

# ----------------------------
# MAIN CHATBOT
# ----------------------------
def run_chatbot(user_input):

    # ✅ STEP 1: Extract symptoms from ORIGINAL text
    symptoms = normalize_input(user_input)

    # ✅ STEP 2: Translate input
    translated_text, lang = to_english(user_input)

    if not translated_text or translated_text.strip() == "":
        translated_text = user_input.lower()

    # ✅ STEP 3: Extract symptoms from translated text ALSO
    symptoms = list(set(symptoms) | set(normalize_input(translated_text)))

    print("🧾 Extracted Symptoms:", symptoms)

    if not symptoms:
        return translate_to_user_lang("Could not understand symptoms.", lang)

    if len(symptoms) < 2:
        return translate_to_user_lang(
            "Please provide more symptoms.", lang
        )

    # ---------------- RULE ENGINE ----------------
    rule = apply_medical_rules(symptoms)

    if rule:
        predictions = [{"disease": d, "confidence": 70 - i * 5} for i, d in enumerate(rule)]
    else:
        predictions = predict(symptoms)

    if not predictions:
        predictions = [{"disease": "General Viral Infection", "confidence": 40}]

    doctor_advice = get_doctor_advice(symptoms, predictions)

    print("DEBUG:", translated_text, symptoms, predictions)

    # ================= GEMINI (FIXED POSITION) =================
    gemini_text = None

    if model_gemini:
        prompt = f"""
You are a friendly medical assistant.

Explain the condition in simple human language in 3-4 lines only.
Do NOT give long answers.

User input: {translated_text}
Symptoms: {', '.join(symptoms)}
Possible diseases: {', '.join([p['disease'] for p in predictions])}
"""
        try:
            gemini_response = model_gemini.generate_content(prompt)

            print("GEMINI RAW:", gemini_response)

            if hasattr(gemini_response, "text") and gemini_response.text:
                gemini_text = gemini_response.text
            else:
                gemini_text = gemini_response.candidates[0].content.parts[0].text

        except Exception as e:
            print("❌ Gemini Error:", e)
            gemini_text = None

    print("GEMINI TEXT:", gemini_text)

    # ================= FINAL RESPONSE =================
    return build_original_response(predictions, doctor_advice, lang, gemini_text)


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    print("💬 AI Health Assistant Ready")

    while True:
        user_input = input("Enter symptoms: ")

        if user_input.lower() == "exit":
            break

        print(run_chatbot(user_input))