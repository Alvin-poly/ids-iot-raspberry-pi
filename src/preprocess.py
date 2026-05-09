import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_DIR = "../data/"

def load_data(data_dir):
    dfs = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            filepath = os.path.join(data_dir, file)
            print(f"Loading: {file}")
            df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
            df.columns = df.columns.str.strip()
            print(f"  -> {len(df)} rows loaded")
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records loaded: {len(combined)}")
    return combined

def clean_data(df):
    print("\nShape before cleaning:", df.shape)

    # Drop non-feature columns
    cols_to_drop = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    print(f"Dropped metadata columns. Remaining columns: {len(df.columns)}")

    # Remove corrupt header rows
    df = df[df["Label"] != "Label"]
    print(f"Removed corrupt header rows. Rows remaining: {len(df)}")

    # Save label column separately
    label_col = df["Label"].copy()

    # Force convert all feature columns to numeric
    df = df.drop("Label", axis=1).apply(pd.to_numeric, errors="coerce")

    # Put label back
    df["Label"] = label_col.values

    # Replace inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop NaN rows
    before = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {before - len(df)} rows with NaN/inf values")

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Dropped {before - len(df)} duplicate rows")

    print(f"\nShape after cleaning: {df.shape}")
    print(f"Dtypes: {df.dtypes.value_counts().to_dict()}")
    return df

def encode_labels(df):
    df["Label"] = df["Label"].apply(
        lambda x: 0 if str(x).strip().lower() == "benign" else 1
    )
    print("\nLabel distribution after encoding:")
    print(df["Label"].value_counts())
    print(f"Attack ratio: {round(df['Label'].mean() * 100, 2)}%")
    return df

def preprocess(df):
    print("\nSeparating features and labels...")

    X = df.drop("Label", axis=1)
    y = df["Label"]

    print(f"Feature count : {X.shape[1]}")
    print(f"Dtypes        : {X.dtypes.value_counts().to_dict()}")

    # Drop rows with any remaining NaN
    before = len(X)
    X.dropna(inplace=True)
    y = y.loc[X.index]
    print(f"Dropped {before - len(X)} remaining NaN rows")

    print(f"\nFinal feature count : {X.shape[1]}")
    print(f"Final row count     : {X.shape[0]}")

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Attack % in train: {round(y_train.mean() * 100, 2)}%")
    print(f"Attack % in test : {round(y_test.mean() * 100, 2)}%")

    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()

# ── Run all steps ──────────────────────────────────────────
df = load_data(DATA_DIR)
df = clean_data(df)
df = encode_labels(df)
X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

# Save scaler and feature names
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(feature_names, "../models/feature_names.pkl")
print("\nScaler and feature names saved to /models/")


# Save preprocessed splits to disk so train.py doesn't redo everything
print("\nSaving preprocessed data to disk...")
np.save("../models/X_train.npy", X_train)
np.save("../models/X_test.npy", X_test)
np.save("../models/y_train.npy", y_train)
np.save("../models/y_test.npy", y_test)
print("Saved X_train, X_test, y_train, y_test to /models/")