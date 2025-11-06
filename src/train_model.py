import json, time, joblib
from pathlib import Path
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

ts = time.strftime("%Y%m%d_%H%M%S")
Path("models").mkdir(exist_ok=True)
Path("metrics").mkdir(exist_ok=True)

data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
metrics = {
    "timestamp": ts,
    "task": "wine_multiclass",
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    "n_classes": int(len(np.unique(y)))
}

model_path = f"models/wine_rf_{ts}.joblib"
joblib.dump(clf, model_path)

with open(f"metrics/metrics_{ts}.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved model: {model_path}")
print("Metrics:", metrics)
