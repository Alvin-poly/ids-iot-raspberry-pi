import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)
import os

# ── Load saved data and model ──────────────────────────────
print("Loading model and test data...")
model         = joblib.load("../models/ids_rf_model.pkl")
X_test        = np.load("../models/X_test.npy")
y_test        = np.load("../models/y_test.npy")
feature_names = joblib.load("../models/feature_names.pkl")

# ── Predictions ────────────────────────────────────────────
print("Generating predictions...")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ── Print metrics ──────────────────────────────────────────
print(f"\nAccuracy : {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
print(f"AUC Score: {round(roc_auc_score(y_test, y_proba), 4)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Benign", "Attack"]))

# ── Create output folder for charts ───────────────────────
os.makedirs("../evaluation", exist_ok=True)

# ── Plot 1: Confusion Matrix ───────────────────────────────
print("Saving confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=["Benign", "Attack"],
    yticklabels=["Benign", "Attack"],
    cmap="Blues"
)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("../evaluation/confusion_matrix.png", dpi=150)
plt.close()
print("Saved: evaluation/confusion_matrix.png")

# ── Plot 2: ROC Curve ──────────────────────────────────────
print("Saving ROC curve...")
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score   = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="steelblue",
         label=f"ROC Curve (AUC = {round(auc_score, 4)})")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../evaluation/roc_curve.png", dpi=150)
plt.close()
print("Saved: evaluation/roc_curve.png")

# ── Plot 3: Precision-Recall Curve ────────────────────────
print("Saving precision-recall curve...")
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, color="darkorange")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("../evaluation/precision_recall_curve.png", dpi=150)
plt.close()
print("Saved: evaluation/precision_recall_curve.png")

# ── Plot 4: Top 20 Feature Importances ────────────────────
print("Saving feature importance chart...")
importances = model.feature_importances_
indices     = np.argsort(importances)[::-1][:20]
top_names   = [feature_names[i] for i in indices]
top_vals    = importances[indices]

plt.figure(figsize=(10, 6))
plt.barh(top_names[::-1], top_vals[::-1], color="steelblue")
plt.xlabel("Importance Score")
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("../evaluation/feature_importance.png", dpi=150)
plt.close()
print("Saved: evaluation/feature_importance.png")

# ── Plot 5: Class Distribution ────────────────────────────
print("Saving class distribution chart...")
labels  = ["Benign", "Attack"]
counts  = [int((y_test == 0).sum()), int((y_test == 1).sum())]
colors  = ["steelblue", "tomato"]

plt.figure(figsize=(6, 4))
plt.bar(labels, counts, color=colors)
for i, count in enumerate(counts):
    plt.text(i, count + 5000, f"{count:,}",
             ha="center", fontsize=11)
plt.title("Test Set Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../evaluation/class_distribution.png", dpi=150)
plt.close()
print("Saved: evaluation/class_distribution.png")

print("\nAll evaluation charts saved to /evaluation/")
print("\nDone!")