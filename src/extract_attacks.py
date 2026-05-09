import pandas as pd
import numpy as np
import joblib
import os

DATA_DIR      = "../data/"
feature_names = joblib.load("../models/feature_names.pkl")
scaler        = joblib.load("../models/scaler.pkl")

all_attacks = []
all_benign  = []

for file in sorted(os.listdir(DATA_DIR)):
    if not file.endswith(".csv"):
        continue
    print(f"Loading {file}...")
    df = pd.read_csv(
        os.path.join(DATA_DIR, file),
        encoding="utf-8",
        low_memory=False
    )
    df.columns = df.columns.str.strip()
    df = df[df["Label"] != "Label"]

    # Separate benign and attack
    benign  = df[df["Label"].str.strip().str.lower() == "benign"]
    attacks = df[df["Label"].str.strip().str.lower() != "benign"]

    all_attacks.append(attacks)
    all_benign.append(benign)

attacks_df = pd.concat(all_attacks, ignore_index=True)
benign_df  = pd.concat(all_benign,  ignore_index=True)

print(f"\nTotal attack rows : {len(attacks_df)}")
print(f"Attack types      : {attacks_df['Label'].unique()}")

# Sample 5 of each attack type and 20 benign
samples = []
for label in attacks_df["Label"].unique():
    subset = attacks_df[attacks_df["Label"] == label].head(5)
    samples.append(subset)

benign_sample = benign_df.head(20)
samples.append(benign_sample)

combined = pd.concat(samples, ignore_index=True)
label_col = combined["Label"].copy()

# Drop non-feature columns
cols_to_drop = ["Flow ID", "Src IP", "Src Port",
                "Dst IP", "Timestamp", "Label"]
combined.drop(
    columns=[c for c in cols_to_drop if c in combined.columns],
    inplace=True
)

# Convert to numeric and select features
combined = combined.apply(pd.to_numeric, errors="coerce")
combined = combined[feature_names]
combined.replace([np.inf, -np.inf], np.nan, inplace=True)
combined.dropna(inplace=True)
label_col = label_col.iloc[:len(combined)]

# Scale
scaled = scaler.transform(combined)

# Save scaled features and labels
np.save("../models/demo_X.npy", scaled)
np.save("../models/demo_labels.npy",
        label_col.values.astype(str))

print(f"\nSaved {len(scaled)} demo rows to /models/")
print("Labels saved:")
unique, counts = np.unique(label_col.values.astype(str),
                           return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u}: {c} rows")