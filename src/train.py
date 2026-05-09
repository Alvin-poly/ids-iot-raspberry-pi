import joblib
import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ── Load saved preprocessed data ──────────────────────────
print("Loading preprocessed data...")
X_train = np.load("../models/X_train.npy")
X_test  = np.load("../models/X_test.npy")
y_train = np.load("../models/y_train.npy")
y_test  = np.load("../models/y_test.npy")

print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"Attack % train: {round(y_train.mean() * 100, 2)}%")
print(f"Attack % test : {round(y_test.mean() * 100, 2)}%")

# ── Train Random Forest ────────────────────────────────────
print("\nTraining Random Forest Classifier...")
print("(This may take several minutes)")

start = time.time()

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
    verbose=1
)

rf.fit(X_train, y_train)

duration = round(time.time() - start, 2)
print(f"\nTraining complete in {duration} seconds")

# ── Evaluate ───────────────────────────────────────────────
print("\nEvaluating on test set...")
y_pred = rf.predict(X_test)

print(f"\nAccuracy : {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Benign", "Attack"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save model ─────────────────────────────────────────────
joblib.dump(rf, "../models/ids_rf_model.pkl")
print("\nModel saved to /models/ids_rf_model.pkl")

size_mb = round(os.path.getsize("../models/ids_rf_model.pkl") / (1024 * 1024), 2)
print(f"Model file size: {size_mb} MB")