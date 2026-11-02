"""Unit tests for Engine 4: Architecture & Toolchain Generator."""
from src.models.schemas import (
    ProblemCategory,
    AnalysisRequest,
    ModelRecommendation,
    TechnicalBlueprint
)
from src.engines.architecture_generator import ArchitectureGenerator


def test_generate_blueprint():
    req = AnalysisRequest(
        problem_description="Build an automated predictive analytics pipeline for customer retention."
    )
    models = [
        ModelRecommendation(
            model_name="XGBoost Classifier",
            tier="Primary Candidate",
            purpose="Gradient boosted decision trees",
            why_suitable="Superior accuracy on tabular features",
            pros=["Fast", "Accurate"],
            cons=["Black box"],
            complexity="Medium",
            hardware_needed="CPU Friendly",
            primary_metrics=["ROC-AUC", "F1"],
            documentation_url="https://xgboost.readthedocs.io/"
        )
    ]
    bp = ArchitectureGenerator.generate_blueprint(
        category=ProblemCategory.TABULAR_CLASSIFICATION,
        category_name="Tabular Classification",
        summary="Customer retention pipeline",
        request=req,
        models=models
    )

    assert isinstance(bp, TechnicalBlueprint)
    assert len(bp.system_architecture) == 6
    assert len(bp.recommended_toolchain) >= 7
    assert len(bp.development_roadmap) == 6
    assert "XGBoost" in bp.starter_code_scaffold
