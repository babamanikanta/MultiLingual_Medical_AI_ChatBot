import pandas as pd
import joblib
import re

from utils.data_loader import get_disease_info
from utils.translator import translate_to_user_lang, to_english
from utils.nlp_extractor import extract_symptoms_nlp

# ----------------------------
# Load model + dataset
# ----------------------------
model = joblib.load("model/disease_model.pkl")

train_df = pd.read_csv("data/training.csv")
train_df.columns = train_df.columns.str.strip().str.replace(" ", "_")

symptom_columns = train_df.columns[:-1].tolist()
ALL_DISEASES = set(model.classes_)
ALL_DISEASES_LOWER = {d.lower() for d in ALL_DISEASES}

# ----------------------------
# Disease Control
# ----------------------------
COMMON_DISEASES = {
    "Common Cold", "Viral Fever", "Flu",
    "Allergy", "Migraine", "Gastroenteritis"
}

SERIOUS_DISEASES = {
    "AIDS", "Paralysis (brain hemorrhage)",
    "Tuberculosis", "Cancer", "Heart attack"
}

# ----------------------------
# SYMPTOM MAP (FINAL)
# ----------------------------
RAW_SYMPTOM_MAP = {

    # English
    "fever": "high_fever",
    "cough": "cough",
    "cold": "continuous_sneezing",
    "headache": "headache",
    "head ache": "headache",
    "stomach pain": "stomach_pain",
    "stomachpain": "stomach_pain",
    "abdominal pain": "abdominal_pain",
    "vomiting": "vomiting",
    "vomitting": "vomiting",
    "vomit": "vomiting",
    "nausea": "nausea",
    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",
    "loose motions": "diarrhoea",

    # Hindi
    "bukhar": "high_fever",
    "khansi": "cough",
    "sir dard": "headache",
    "sar dard": "headache",
    "pet dard": "stomach_pain",
    "pet mein dard": "stomach_pain",
    "ulti": "vomiting",
    "ulti ho rahi": "vomiting",

    # Telugu (roman)
    "jwaram": "high_fever",
    "daggu": "cough",
    "tala noppi": "headache",
    "kadupu noppi": "stomach_pain",
    "vanti": "vomiting",

    # Telugu (native)
    "జ్వరం": "high_fever",
    "దగ్గు": "cough",
    "తలనొప్పి": "headache",
    "కడుపు నొప్పి": "stomach_pain",
    "వాంతులు": "vomiting",

    # Runny nose (FIXED STRONG)
    "mukku karutundi": "runny_nose",
    "mukku nundi neeru vastundi": "runny_nose",
    "neeru vastundi": "runny_nose",
    "cheemidi karutundi": "runny_nose",
    "చెమిడి కారుతుంది": "runny_nose",
    "చెమిడి": "runny_nose",
    "ముక్కు కారుతుంది": "runny_nose",

    # Cold
    "jalubu": "continuous_sneezing",
    "జలుబు": "continuous_sneezing",

    # Diarrhoea Telugu
    "dayeriya": "diarrhoea",
    "డయేరియా": "diarrhoea",
}

SYMPTOM_MAP = {k: v for k, v in RAW_SYMPTOM_MAP.items() if v in symptom_columns}

# ----------------------------
# NORMALIZATION
# ----------------------------
def normalize_input(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)

    symptoms = set()

    # 🔥 STEP 1: NLP extraction
    symptoms.update(extract_symptoms_nlp(text, symptom_columns))

    # 🔥 STEP 2: rule mapping
    for key, value in SYMPTOM_MAP.items():
        if key in text:
            symptoms.add(value)

    # 🔥 STEP 3: FIX "aur / and" splitting (CRITICAL FIX)
    parts = re.split(r"\baur\b|\band\b", text)

    for part in parts:
        part = part.strip()

        for key, value in SYMPTOM_MAP.items():
            if key in part:
                symptoms.add(value)

    # 🔥 STEP 4: direct column match
    for col in symptom_columns:
        if col.replace("_", " ") in text:
            symptoms.add(col)

    return list({s for s in symptoms if s in symptom_columns})

# ----------------------------
# RULE ENGINE
# ----------------------------
def apply_medical_rules(symptoms):
    s = set(symptoms)

    # Prevent serious misclassification
    if "runny_nose" in s or "continuous_sneezing" in s:
        return ["Common Cold", "Allergy"]

    if "high_fever" in s and "cough" in s:
        return ["Viral Fever", "Flu"]

    if "headache" in s and "vomiting" in s:
        return ["Migraine"]

    if ("stomach_pain" in s or "abdominal_pain" in s) and "diarrhoea" in s:
        return ["Gastroenteritis"]

    if "high_fever" in s and "headache" in s:
        return ["Viral Fever"]

    if "chest_pain" in s and "breathlessness" in s:
        return ["Heart attack"]

    return []

# ----------------------------
# ML PREDICTION (SAFE)
# ----------------------------
def predict(symptoms_list):
    input_data = pd.DataFrame([[0]*len(symptom_columns)], columns=symptom_columns)

    for s in symptoms_list:
        if s in symptom_columns:
            input_data.loc[0, s] = 1

    probs = model.predict_proba(input_data)[0]
    classes = model.classes_

    results = []

    for i in range(len(classes)):
        disease = classes[i]
        confidence = probs[i] * 100

        # 🚫 Heart attack safety
        if disease == "Heart attack":
            if not any(x in symptoms_list for x in ["chest_pain", "breathlessness"]):
                continue

        # 🚫 Block serious diseases
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
# DOCTOR RECOMMENDATION
# ----------------------------
def get_doctor_advice(symptoms, predictions):
    s = set(symptoms)

    if "chest_pain" in s or "breathlessness" in s:
        return "🚨 Seek immediate medical attention immediately."

    if "high_fever" in s:
        return "⚠️ If fever lasts more than 2-3 days, consult a doctor."

    if "vomiting" in s and "diarrhoea" in s:
        return "⚠️ Risk of dehydration. Consult a doctor if symptoms persist."

    if predictions and predictions[0]["confidence"] < 60:
        return "⚠️ Diagnosis is uncertain. Please consult a doctor."

    return "✅ Monitor symptoms. Consult a doctor if condition worsens."

# ----------------------------
# MAIN CHATBOT
# ----------------------------
def run_chatbot(user_input):

    # 🔥 STEP 1: TRANSLATE FIRST (VERY IMPORTANT FIX)
    translated_text, lang = to_english(user_input)

    # fallback safety
    if not translated_text:
        translated_text = user_input.lower()

    # 🔥 STEP 2: normalize + extract symptoms on ENGLISH TEXT
    symptoms = normalize_input(translated_text.lower())

    print("🧾 Extracted Symptoms:", symptoms)

    if not symptoms:
        return translate_to_user_lang(
            "Could not understand symptoms.",
            lang
        )

    if len(symptoms) < 2:
        return translate_to_user_lang(
            "I understood limited symptoms. Please add more details like fever, pain, cough etc.",
            lang
        )

    # 🔥 STEP 3: rule engine
    rule = apply_medical_rules(symptoms)

    if rule:
        predictions = [
            {"disease": d, "confidence": 70 - i * 5}
            for i, d in enumerate(rule)
        ]
    else:
        predictions = predict(symptoms)

    if not predictions:
        predictions = [{
            "disease": "General Viral Infection",
            "confidence": 40
        }]

    doctor_advice = get_doctor_advice(symptoms, predictions)

    # 🔥 STEP 4: build response
    response = ""

    for p in predictions:
        info = get_disease_info(p["disease"])

        response += f"""
🦠 Disease: {p['disease']}
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

-------------------------
"""

    response += f"""

🩺 Doctor Recommendation:
{doctor_advice}
"""

    # 🔥 STEP 5: translate back
    return translate_to_user_lang(response, lang)

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    print("💬 AI Health Assistant Ready (FINAL PRODUCTION SAFE)")

    while True:
        user_input = input("Enter symptoms: ")

        if user_input.lower() == "exit":
            break

        print(run_chatbot(user_input))