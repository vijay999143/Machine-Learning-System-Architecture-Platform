"""Unit tests for Engine 3: Model Recommender."""
from src.models.schemas import (
    ProblemCategory,
    AnalysisRequest,
    HardwareTarget,
    InterpretabilityRequirement,
    LatencyRequirement,
    DatasetProfile
)
from src.engines.model_recommender import ModelRecommender


def test_recommend_tabular_classification():
    req = AnalysisRequest(
        problem_description="Customer churn prediction with high interpretability",
        interpretability=InterpretabilityRequirement.HIGH,
        hardware=HardwareTarget.CPU_ONLY,
        latency=LatencyRequirement.REAL_TIME
    )
    models = ModelRecommender.recommend_models(ProblemCategory.TABULAR_CLASSIFICATION, req)

    assert len(models) >= 2
    tiers = [m.tier for m in models]
    assert any("Primary" in t or "Baseline" in t for t in tiers)
    for m in models:
        assert m.model_name
        assert len(m.pros) > 0
        assert m.documentation_url.startswith("http")


def test_recommend_with_dataset_profile():
    req = AnalysisRequest(
        problem_description="Credit card fraud detection",
        hardware=HardwareTarget.CPU_ONLY
    )
    profile = DatasetProfile(
        num_rows=50000,
        num_cols=30,
        memory_usage_mb=12.5,
        numeric_features=["v1", "v2"],
        categorical_features=[],
        datetime_features=[],
        text_features=[],
        missing_value_summary={},
        detected_target="is_fraud",
        class_imbalance_ratio=15.0,
        recommended_preprocessing=["Apply SMOTE"]
    )
    models = ModelRecommender.recommend_models(ProblemCategory.ANOMALY_DETECTION, req, profile)
    assert len(models) >= 2
    # Check that imbalance was incorporated into why_suitable
    combined_reasons = " ".join([m.why_suitable for m in models])
    assert "imbalance" in combined_reasons.lower()
