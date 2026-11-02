"""Benchmark quality tests for ML/DL Model Recommendation Engine across real-world problem domains."""
import pytest
from src.models.schemas import (
    ProblemCategory,
    AnalysisRequest,
    LatencyRequirement,
    HardwareTarget,
    InterpretabilityRequirement
)
from src.engines.problem_analyzer import ProblemAnalyzer
from src.engines.model_recommender import ModelRecommender

BENCHMARK_CASES = [
    {
        "domain": "Credit Card Fraud",
        "description": "Detect fraudulent transactions on credit cards with extreme imbalance and real-time latency",
        "expected_category": [ProblemCategory.TABULAR_CLASSIFICATION, ProblemCategory.ANOMALY_DETECTION],
        "latency": LatencyRequirement.REAL_TIME,
        "hardware": HardwareTarget.CPU_ONLY,
        "interpretability": InterpretabilityRequirement.HIGH
    },
    {
        "domain": "Telecom Churn",
        "description": "Predict whether customer will cancel subscription based on tenure and billing data",
        "expected_category": [ProblemCategory.TABULAR_CLASSIFICATION],
        "latency": LatencyRequirement.NEAR_REAL_TIME,
        "hardware": HardwareTarget.CPU_ONLY,
        "interpretability": InterpretabilityRequirement.MEDIUM
    },
    {
        "domain": "Chest X-Ray Pneumonia",
        "description": "Classify chest radiograph images to identify viral pneumonia vs bacterial pneumonia",
        "expected_category": [ProblemCategory.CV_IMAGE_CLASSIFICATION],
        "latency": LatencyRequirement.NEAR_REAL_TIME,
        "hardware": HardwareTarget.SINGLE_GPU,
        "interpretability": InterpretabilityRequirement.HIGH
    },
    {
        "domain": "Pedestrian Detection",
        "description": "Real-time object detection and bounding box localization of pedestrians from video streams using YOLO",
        "expected_category": [ProblemCategory.CV_OBJECT_DETECTION],
        "latency": LatencyRequirement.REAL_TIME,
        "hardware": HardwareTarget.SINGLE_GPU,
        "interpretability": InterpretabilityRequirement.LOW
    },
    {
        "domain": "Customer Review Sentiment",
        "description": "Sentiment analysis and topic classification of customer product feedback reviews",
        "expected_category": [ProblemCategory.NLP_TEXT_CLASSIFICATION],
        "latency": LatencyRequirement.NEAR_REAL_TIME,
        "hardware": HardwareTarget.CPU_ONLY,
        "interpretability": InterpretabilityRequirement.MEDIUM
    },
    {
        "domain": "Retail Store Sales",
        "description": "Forecast daily supermarket sales and demand over the next 30 days with holidays",
        "expected_category": [ProblemCategory.TIME_SERIES],
        "latency": LatencyRequirement.BATCH,
        "hardware": HardwareTarget.CPU_ONLY,
        "interpretability": InterpretabilityRequirement.HIGH
    },
    {
        "domain": "Machine Vibration Anomaly",
        "description": "Unsupervised anomaly detection for industrial turbine vibration sensors without labels",
        "expected_category": [ProblemCategory.ANOMALY_DETECTION],
        "latency": LatencyRequirement.REAL_TIME,
        "hardware": HardwareTarget.CPU_ONLY,
        "interpretability": InterpretabilityRequirement.MEDIUM
    },
    {
        "domain": "Movie Recommender",
        "description": "Build a collaborative filtering movie recommendation system for users based on ratings",
        "expected_category": [ProblemCategory.RECOMMENDATION_SYSTEMS],
        "latency": LatencyRequirement.REAL_TIME,
        "hardware": HardwareTarget.CPU_ONLY,
        "interpretability": InterpretabilityRequirement.LOW
    }
]


@pytest.mark.parametrize("case", BENCHMARK_CASES)
def test_recommendation_benchmark_cases(case):
    """Ensure each benchmark case classifies correctly and produces tiered candidates."""
    req = AnalysisRequest(
        problem_description=case["description"],
        latency=case["latency"],
        hardware=case["hardware"],
        interpretability=case["interpretability"]
    )
    category, display_name, summary = ProblemAnalyzer.analyze_problem(req)
    assert category in case["expected_category"]
    assert display_name

    models = ModelRecommender.recommend_models(category, req)
    assert len(models) >= 2

    # Verify model recommendation properties
    for model in models:
        assert model.model_name
        assert model.tier
        assert len(model.pros) > 0
        assert len(model.primary_metrics) > 0
