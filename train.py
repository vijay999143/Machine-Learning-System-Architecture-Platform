import json
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
METRICS_PATH = Path(__file__).resolve().parent / "metrics.json"


def train_model():
    """Train Random Forest model on the Iris dataset and save artifacts."""
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = list(iris.target_names)
    feature_names = list(iris.feature_names)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted"))
    recall = float(recall_score(y_test, y_pred, average="weighted"))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    # Save model and metadata
    joblib.dump(
        {
            "model": model,
            "target_names": target_names,
            "feature_names": feature_names
        },
        MODEL_PATH
    )

    # Save evaluation metrics
    metrics_data = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": conf_matrix,
        "target_names": target_names,
        "feature_names": feature_names,
        "test_samples": len(y_test)
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print("Model Training Completed")
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model Precision: {precision:.2f}")
    print(f"Model Recall: {recall:.2f}")
    print(f"Model F1-Score: {f1:.2f}")

    return accuracy


if __name__ == "__main__":
    train_model()

