import joblib
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(script_dir, "disease_model.pkl"))

# Load columns (symptoms list)
training_df = pd.read_csv(os.path.join(script_dir, "../data/training.csv"))
symptom_columns = training_df.columns[:-1]


def predict_disease(user_input):
    user_symptoms = [s.strip().lower().replace(" ", "_") for s in user_input.split()]

    input_data = [0] * len(symptom_columns)

    for symptom in user_symptoms:
        if symptom in symptom_columns:
            idx = list(symptom_columns).index(symptom)
            input_data[idx] = 1

    input_df = pd.DataFrame([input_data], columns=symptom_columns)

    prediction = model.predict(input_df)[0]
    probability = max(model.predict_proba(input_df)[0]) * 100

    return [{
        "disease": prediction,
        "confidence": round(probability, 2)
    }]