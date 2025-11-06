import json, joblib
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load Wine dataset (same as training)
data = load_wine()
X, y = data.data, data.target
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Get latest model saved in models/
model_files = sorted(Path("models").glob("*.joblib"))
latest_model = model_files[-1]

clf = joblib.load(latest_model)
y_pred = clf.predict(X_test)

# Save new evaluation metrics
metrics = {
    "evaluated_model": latest_model.name,
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
}

with open("metrics/eval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Evaluation complete:", metrics)
