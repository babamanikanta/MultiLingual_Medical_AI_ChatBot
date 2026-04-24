import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("../data/training_improved.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

severity_df = pd.read_csv("../data/symptom-severity.csv")
severity_map = dict(zip(severity_df["Symptom"], severity_df["weight"]))

# Apply weights
for col in df.columns[:-1]:
    if col in severity_map:
        df[col] = df[col] * severity_map[col]

# Features & Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,          # 🔥 increased for realism
    random_state=42,
    stratify=y
)

# ----------------------------
# Model (ANTI-OVERFITTING)
# ----------------------------
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,            # 🔥 LIMIT DEPTH
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Test Accuracy:", round(accuracy * 100, 2), "%")

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# Cross Validation
# ----------------------------
cv_scores = cross_val_score(model, X, y, cv=5)

print("\n🔁 Cross Validation Scores:", cv_scores)
print("📈 Average CV Accuracy:", round(cv_scores.mean() * 100, 2), "%")

# ----------------------------
# Save model
# ----------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/disease_model.pkl")

print("\n💾 Model saved at: model/disease_model.pkl")