import json, time, joblib, sys
from pathlib import Path
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss

Path("models").mkdir(exist_ok=True)
Path("metrics").mkdir(exist_ok=True)

model_files = sorted(Path("models").glob("*.joblib"))
if not model_files:
    import train_model
    model_files = sorted(Path("models").glob("*.joblib"))
    if not model_files:
        sys.exit(1)

latest_model = model_files[-1]

data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

base_model = joblib.load(latest_model)

calib = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
calib.fit(X_train, y_train)

proba = calib.predict_proba(X_test)
pred = np.argmax(proba, axis=1)
metrics = {
    "base_model": latest_model.name,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "accuracy": float(accuracy_score(y_test, pred)),
    "f1_macro": float(f1_score(y_test, pred, average="macro")),
    "brier_macro": float(np.mean([
        brier_score_loss((y_test == k).astype(int), proba[:, k])
        for k in np.unique(y)
    ]))
}

cal_ts = metrics["timestamp"]
calib_path = f"models/{latest_model.stem}_calibrated_{cal_ts}.joblib"
joblib.dump(calib, calib_path)

with open(f"metrics/calibrated_metrics_{cal_ts}.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved calibrated model:", calib_path)
print("Calibration metrics:", metrics)
