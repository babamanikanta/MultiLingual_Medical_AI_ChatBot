import pandas as pd
import random

# Load original dataset
df = pd.read_csv("data/training.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

new_rows = []

for _, row in df.iterrows():
    symptoms = row[:-1].values.copy()
    disease = row[-1]

    active_indices = [i for i, val in enumerate(symptoms) if val == 1]
    all_indices = list(range(len(symptoms)))

    # 🔁 Generate more realistic variations
    for _ in range(5):   # increase variations

        temp = symptoms.copy()

        # 🔻 REMOVE symptoms (simulate incomplete input)
        remove_count = random.randint(1, 2)
        for _ in range(remove_count):
            if len(active_indices) > 2:
                remove_idx = random.choice(active_indices)
                temp[remove_idx] = 0

        # 🔺 ADD noise symptoms (simulate confusion)
        add_count = random.randint(0, 2)
        for _ in range(add_count):
            add_idx = random.choice(all_indices)
            temp[add_idx] = 1

        new_rows.append(list(temp) + [disease])

# Create new dataset
new_df = pd.DataFrame(new_rows, columns=df.columns)

# Combine original + noisy data
final_df = pd.concat([df, new_df]).sample(frac=1).reset_index(drop=True)

# Save
final_df.to_csv("data/training_improved.csv", index=False)

print("✅ Improved dataset created:", final_df.shape)