"""Tests for AnomalyDetector"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import tempfile
import shutil
from src.ml.anomaly_detector import AnomalyDetector


def _sample_features(n=500):
    np.random.seed(42)
    normal = np.random.randn(n, 5)
    # Inject some outliers
    outliers = np.random.randn(int(n * 0.05), 5) * 10 + 5
    data = np.vstack([normal, outliers])
    return pd.DataFrame(data, columns=[f"feat_{i}" for i in range(5)])


def test_train():
    detector = AnomalyDetector()
    features = _sample_features()
    stats = detector.train(features)
    assert detector.is_fitted
    assert "n_samples" in stats
    assert "n_anomalies" in stats
    assert stats["n_anomalies"] > 0


def test_predict():
    detector = AnomalyDetector()
    features = _sample_features()
    detector.train(features)
    results = detector.predict(features)
    assert "is_anomaly" in results.columns
    assert "anomaly_score" in results.columns
    assert "anomaly_severity" in results.columns
    assert results["is_anomaly"].isin([0, 1]).all()


def test_summary():
    detector = AnomalyDetector()
    features = _sample_features()
    detector.train(features)
    results = detector.predict(features)
    summary = detector.get_anomaly_summary(results)
    assert summary["total_records"] == len(results)
    assert "anomaly_rate" in summary


def test_save_load():
    detector = AnomalyDetector()
    features = _sample_features()
    detector.train(features)

    tmp_dir = tempfile.mkdtemp()
    try:
        detector.save_model(tmp_dir)

        # Load in a new instance
        new_detector = AnomalyDetector()
        assert new_detector.load_model(tmp_dir)
        assert new_detector.is_fitted

        # Predict with loaded model
        results = new_detector.predict(features)
        assert "is_anomaly" in results.columns
    finally:
        shutil.rmtree(tmp_dir)


def test_no_predict_before_train():
    detector = AnomalyDetector()
    features = _sample_features()
    try:
        detector.predict(features)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
