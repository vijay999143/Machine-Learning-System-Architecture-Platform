"""Unit tests for Engine 2: Dataset Analyzer."""
import pandas as pd
import numpy as np
from src.engines.dataset_analyzer import DatasetAnalyzer


def test_dataset_profiler():
    # Construct synthetic DataFrame
    data = {
        "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "age": [25, 45, 32, np.nan, 50, 23, 67, 34, 29, 41],
        "gender": ["M", "F", "F", "M", "F", "M", "M", "F", "M", "F"],
        "balance": [1200.5, 450.0, 9800.0, 2100.0, 50.0, 310.0, 15000.0, 890.0, 4200.0, 670.0],
        "churn": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # Imbalanced 8:2
    }
    df = pd.DataFrame(data)

    profile = DatasetAnalyzer.profile_dataframe(df)

    assert profile.num_rows == 10
    assert profile.num_cols == 5
    assert "age" in profile.missing_value_summary
    assert "gender" in profile.categorical_features
    assert "balance" in profile.numeric_features
    assert profile.detected_target == "churn"
    assert profile.target_type == "binary_classification"
    assert profile.class_imbalance_ratio == 4.0
    assert len(profile.recommended_preprocessing) >= 2


def test_clean_dataset_no_missing():
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0],
        "feature_2": [4.0, 5.0, 6.0],
        "target": [0, 1, 0]
    })
    profile = DatasetAnalyzer.profile_dataframe(df)
    assert len(profile.missing_value_summary) == 0
    assert profile.detected_target == "target"
