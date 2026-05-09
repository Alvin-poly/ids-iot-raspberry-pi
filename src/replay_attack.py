import joblib
import numpy as np
import pandas as pd
import os

DATA_DIR = "../data/"
model         = joblib.load("../models/ids_rf_model.pkl")
scaler        = joblib.load("../models/scaler.pkl")
feature_names = joblib.load("../models/feature_names.pkl")
model.verbose = 0

# Load one CSV and find real attack rows
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        print(f"Loading {file}...")
        df = pd.read_csv(
            os.path.join(DATA_DIR, file),
            encoding="utf-8",
            low_memory=False
        )
        df.columns = df.columns.str.strip()

        # Find attack rows
        attacks = df[df["Label"] != "Benign"]
        if len(attacks) > 0:
            print(f"Found {len(attacks)} attack rows in {file}")
            print(f"Attack types: {attacks['Label'].unique()}")

            # Take first 5 attack rows
            sample = attacks.head(5).copy()
            label_col = sample["Label"].copy()

            # Drop non-feature columns
            cols_to_drop = ["Flow ID", "Src IP", "Src Port",
                           "Dst IP", "Timestamp", "Label"]
            sample.drop(columns=[c for c in cols_to_drop
                                 if c in sample.columns], inplace=True)

            # Convert to numeric
            sample = sample.apply(pd.to_numeric, errors="coerce")
            sample = sample[feature_names]
            sample.replace([np.inf, -np.inf], np.nan, inplace=True)
            sample.dropna(inplace=True)

            if len(sample) == 0:
                continue

            # Scale and predict
            scaled = scaler.transform(sample)
            preds  = model.predict(scaled)
            probas = model.predict_proba(scaled)[:, 1]

            print("\nResults on real attack rows:")
            for i, (pred, proba, label) in enumerate(
                    zip(preds, probas, label_col)):
                result = "ATTACK ✓" if pred == 1 else "Missed ✗"
                print(f"  Row {i+1} [{label}]: "
                      f"{result} | confidence: {round(proba*100,2)}%")
            break