"""Integration tests for FastAPI endpoints."""
from pathlib import Path
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["catalog_size"] > 15


def test_catalog():
    res = client.get("/api/catalog")
    assert res.status_code == 200
    data = res.json()
    assert "catalog" in data
    assert "xgboost_classifier" in data["catalog"]


def test_presets():
    res = client.get("/api/presets")
    assert res.status_code == 200
    presets = res.json()["presets"]
    assert len(presets) >= 4


def test_analyze_problem_endpoint():
    payload = {
        "problem_description": "We need to predict customer churn in a telecommunications subscription business.",
        "expected_output": "Probability of churn",
        "latency": "near_real_time",
        "hardware": "cpu_only",
        "interpretability": "medium"
    }
    res = client.post("/api/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "project_title" in data
    assert len(data["recommended_models"]) >= 2
    assert len(data["system_architecture"]) == 6
    assert len(data["recommended_toolchain"]) >= 7


def test_analyze_dataset_endpoint():
    sample_csv = Path(__file__).resolve().parent.parent / "sample_datasets" / "sample_credit_card_fraud.csv"
    assert sample_csv.exists()

    with open(sample_csv, "rb") as f:
        files = {"file": ("sample_credit_card_fraud.csv", f, "text/csv")}
        data = {
            "problem_description": "Detect fraudulent credit card charges from tabular stream",
            "latency": "real_time",
            "hardware": "cpu_only",
            "interpretability": "high"
        }
        res = client.post("/api/analyze-dataset", files=files, data=data)

    assert res.status_code == 200
    result = res.json()
    assert result["dataset_profile"] is not None
    assert result["dataset_profile"]["num_rows"] == 20
    assert result["dataset_profile"]["detected_target"] == "is_fraud"
    assert len(result["recommended_models"]) >= 2


def test_dashboard_ui():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "OmniArchitect AI" in res.text

