import pandas as pd
import os

base_path = "data"

description = pd.read_csv(os.path.join(base_path, "description.csv"))
medications = pd.read_csv(os.path.join(base_path, "medications.csv"))
diets = pd.read_csv(os.path.join(base_path, "diets.csv"))
precautions = pd.read_csv(os.path.join(base_path, "precautions_df.csv"))
workout = pd.read_csv(os.path.join(base_path, "workout_df.csv"))


# ----------------------------
# SAFE LOOKUP FUNCTION
# ----------------------------
def safe_get(df, disease, column, default):
    try:
        row = df[df["Disease"] == disease]
        if not row.empty:
            value = row.iloc[0][column]
            if pd.isna(value) or value == "":
                return default
            return value
    except:
        pass
    return default


# ----------------------------
# MAIN FUNCTION
# ----------------------------
def get_disease_info(disease):

    description_text = safe_get(
        description, disease, "Description",
        "No description available."
    )

    medication = safe_get(
        medications, disease, "Medication",
        "Consult a doctor"
    )

    diet = safe_get(
        diets, disease, "Diet",
        "Balanced diet recommended"
    )

    # precautions (multiple columns)
    try:
        row = precautions[precautions["Disease"] == disease]
        if not row.empty:
            precautions_list = row.iloc[0, 1:].dropna().tolist()
        else:
            precautions_list = ["Consult doctor", "Stay hydrated"]
    except:
        precautions_list = ["Consult doctor"]

    # workout
    try:
        workout_list = workout[workout["disease"] == disease]["workout"].tolist()
        if not workout_list:
            workout_list = ["Light exercise", "Stay active"]
    except:
        workout_list = ["Light exercise"]

    return {
        "description": description_text,
        "medication": medication,
        "diet": diet,
        "precautions": precautions_list,
        "workout": workout_list
    }