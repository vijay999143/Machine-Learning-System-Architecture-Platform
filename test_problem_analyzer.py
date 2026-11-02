"""Unit tests for Engine 1: Problem Analyzer."""
import pytest
from src.models.schemas import (
    ProblemCategory,
    AnalysisRequest,
    LatencyRequirement,
    HardwareTarget,
    InterpretabilityRequirement
)
from src.engines.problem_analyzer import ProblemAnalyzer


def test_fraud_detection_classification():
    req = AnalysisRequest(
        problem_description="Develop an automated credit card fraud detection system on transaction streams.",
        expected_output="Binary score 0 or 1",
        latency=LatencyRequirement.REAL_TIME,
        hardware=HardwareTarget.CPU_ONLY,
        interpretability=InterpretabilityRequirement.HIGH
    )
    cat, name, summary = ProblemAnalyzer.analyze_problem(req)
    assert cat in [ProblemCategory.TABULAR_CLASSIFICATION, ProblemCategory.ANOMALY_DETECTION]
    assert len(summary) > 20


def test_computer_vision_classification():
    req = AnalysisRequest(
        problem_description="Classify medical chest x-ray images to identify viral pneumonia vs normal lung tissue.",
        latency=LatencyRequirement.NEAR_REAL_TIME,
        hardware=HardwareTarget.SINGLE_GPU
    )
    cat, name, summary = ProblemAnalyzer.analyze_problem(req)
    assert cat == ProblemCategory.CV_IMAGE_CLASSIFICATION
    assert "Computer Vision" in name


def test_object_detection():
    req = AnalysisRequest(
        problem_description="Real-time object detection and bounding box localization of pedestrians and vehicles using YOLO.",
        latency=LatencyRequirement.REAL_TIME,
        hardware=HardwareTarget.SINGLE_GPU
    )
    cat, name, summary = ProblemAnalyzer.analyze_problem(req)
    assert cat == ProblemCategory.CV_OBJECT_DETECTION
    assert "Object Detection" in name


def test_nlp_sentiment_analysis():
    req = AnalysisRequest(
        problem_description="Perform sentiment analysis and text classification on customer product reviews.",
        latency=LatencyRequirement.NEAR_REAL_TIME
    )
    cat, name, summary = ProblemAnalyzer.analyze_problem(req)
    assert cat == ProblemCategory.NLP_TEXT_CLASSIFICATION


def test_time_series_forecasting():
    req = AnalysisRequest(
        problem_description="Forecast future daily electricity demand and sales trends for the next 30 days.",
        latency=LatencyRequirement.BATCH
    )
    cat, name, summary = ProblemAnalyzer.analyze_problem(req)
    assert cat == ProblemCategory.TIME_SERIES


def test_unsupervised_clustering():
    req = AnalysisRequest(
        problem_description="Perform customer segmentation and unsupervised grouping using transaction behaviors.",
        latency=LatencyRequirement.BATCH
    )
    cat, name, summary = ProblemAnalyzer.analyze_problem(req)
    assert cat == ProblemCategory.TABULAR_CLUSTERING
