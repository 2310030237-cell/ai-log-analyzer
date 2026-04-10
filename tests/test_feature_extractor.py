"""Tests for FeatureExtractor"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.processing.feature_extractor import FeatureExtractor


def _sample_df():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=100, freq="h"),
        "level": np.random.choice(["INFO", "WARNING", "ERROR", "CRITICAL"], 100, p=[0.6, 0.2, 0.15, 0.05]),
        "message": [f"Sample log message {i} with error or success" for i in range(100)],
        "severity_num": np.random.choice([1, 2, 3, 4], 100),
        "service": np.random.choice(["auth", "api", "payment"], 100),
    })


def test_record_features_shape():
    df = _sample_df()
    extractor = FeatureExtractor()
    features = extractor.extract_features(df, mode="record")
    assert len(features) == len(df)
    assert features.shape[1] > 5  # Should have multiple features


def test_record_features_no_nan():
    df = _sample_df()
    extractor = FeatureExtractor()
    features = extractor.extract_features(df, mode="record")
    assert features.isnull().sum().sum() == 0


def test_window_features():
    df = _sample_df()
    extractor = FeatureExtractor()
    features = extractor.extract_features(df, mode="window")
    assert len(features) > 0
    assert "log_count" in features.columns


def test_time_features():
    df = _sample_df()
    extractor = FeatureExtractor()
    features = extractor.extract_features(df, mode="record")
    assert "hour" in features.columns
    assert "day_of_week" in features.columns
    assert "is_weekend" in features.columns
    assert features["hour"].between(0, 23).all()


def test_message_features():
    df = _sample_df()
    extractor = FeatureExtractor()
    features = extractor.extract_features(df, mode="record")
    assert "message_length" in features.columns
    assert "word_count" in features.columns
    assert "has_error_keyword" in features.columns


def test_feature_names():
    df = _sample_df()
    extractor = FeatureExtractor()
    features = extractor.extract_features(df, mode="record")
    assert len(extractor.feature_names) == features.shape[1]
