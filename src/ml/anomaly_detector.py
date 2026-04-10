"""
Anomaly Detection Module
Uses Isolation Forest for unsupervised anomaly detection on log data features.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector for log data.
    Detects unusual patterns in log features that may indicate
    security threats, system failures, or configuration issues.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        ml_config = self.config.get("ml", {}).get("anomaly_detection", {})

        self.n_estimators = ml_config.get("n_estimators", 200)
        self.contamination = ml_config.get("contamination", 0.05)
        self.random_state = ml_config.get("random_state", 42)

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.training_stats = {}

    def train(self, features_df: pd.DataFrame) -> Dict:
        """
        Train the Isolation Forest model on feature data.

        Args:
            features_df: DataFrame of numerical features

        Returns:
            Training statistics dictionary
        """
        print("\n" + "=" * 60)
        print("ANOMALY DETECTION - TRAINING")
        print("=" * 60)

        # Select only numeric columns
        numeric_df = features_df.select_dtypes(include=[np.number])
        self.feature_names = list(numeric_df.columns)

        print(f"  Features: {len(self.feature_names)}")
        print(f"  Samples: {len(numeric_df):,}")
        print(f"  Algorithm: Isolation Forest")
        print(f"  Estimators: {self.n_estimators}")
        print(f"  Contamination: {self.contamination}")

        # Handle infinities and NaN
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Scale features
        X_scaled = self.scaler.fit_transform(numeric_df)

        # Train Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )

        print("\n  Training model...")
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Get training predictions for statistics
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        n_anomalies = (predictions == -1).sum()
        self.training_stats = {
            "n_samples": len(numeric_df),
            "n_features": len(self.feature_names),
            "n_anomalies": int(n_anomalies),
            "anomaly_rate": round(n_anomalies / len(numeric_df) * 100, 2),
            "score_mean": round(float(scores.mean()), 4),
            "score_std": round(float(scores.std()), 4),
            "score_min": round(float(scores.min()), 4),
            "score_max": round(float(scores.max()), 4),
            "trained_at": datetime.now().isoformat(),
            "feature_names": self.feature_names,
        }

        print(f"\n  [OK] Training complete!")
        print(f"    Anomalies detected: {n_anomalies:,} ({self.training_stats['anomaly_rate']}%)")
        print(f"    Score range: [{self.training_stats['score_min']}, {self.training_stats['score_max']}]")

        return self.training_stats

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies in new data.

        Args:
            features_df: DataFrame of numerical features

        Returns:
            DataFrame with anomaly labels and scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        numeric_df = features_df.select_dtypes(include=[np.number])

        # Align columns with training features
        missing_cols = set(self.feature_names) - set(numeric_df.columns)
        for col in missing_cols:
            numeric_df[col] = 0
        numeric_df = numeric_df[self.feature_names]

        # Handle infinities and NaN
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Scale and predict
        X_scaled = self.scaler.transform(numeric_df)
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        # Build results
        results = features_df.copy()
        results["anomaly_label"] = predictions  # 1 = normal, -1 = anomaly
        results["is_anomaly"] = (predictions == -1).astype(int)
        results["anomaly_score"] = scores

        # Classify severity based on score
        results["anomaly_severity"] = pd.cut(
            scores,
            bins=[-np.inf, -0.3, -0.1, 0, np.inf],
            labels=["critical", "high", "medium", "normal"]
        )

        return results

    def get_anomaly_summary(self, results_df: pd.DataFrame) -> Dict:
        """Generate a summary of detected anomalies."""
        anomalies = results_df[results_df["is_anomaly"] == 1]

        summary = {
            "total_records": len(results_df),
            "total_anomalies": len(anomalies),
            "anomaly_rate": round(len(anomalies) / max(len(results_df), 1) * 100, 2),
            "severity_distribution": {},
            "score_stats": {},
        }

        if "anomaly_severity" in anomalies.columns and len(anomalies) > 0:
            severity_counts = anomalies["anomaly_severity"].value_counts().to_dict()
            summary["severity_distribution"] = {str(k): int(v) for k, v in severity_counts.items()}

        if "anomaly_score" in results_df.columns:
            summary["score_stats"] = {
                "mean": round(float(results_df["anomaly_score"].mean()), 4),
                "std": round(float(results_df["anomaly_score"].std()), 4),
                "anomaly_threshold": round(float(anomalies["anomaly_score"].max()), 4) if len(anomalies) > 0 else 0,
            }

        # Top anomalous features
        if len(anomalies) > 0:
            normal = results_df[results_df["is_anomaly"] == 0]
            feature_diffs = {}
            for col in self.feature_names:
                if col in anomalies.columns and col in normal.columns:
                    anom_mean = anomalies[col].mean()
                    norm_mean = normal[col].mean()
                    if norm_mean != 0:
                        feature_diffs[col] = round(abs(anom_mean - norm_mean) / abs(norm_mean), 4)
            # Top 10 most different features
            top_features = sorted(feature_diffs.items(), key=lambda x: x[1], reverse=True)[:10]
            summary["top_anomalous_features"] = dict(top_features)

        return summary

    def save_model(self, output_dir: str = "data/models") -> str:
        """Save the trained model and scaler to disk."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "anomaly_detector.joblib")
        scaler_path = os.path.join(output_dir, "anomaly_scaler.joblib")
        meta_path = os.path.join(output_dir, "anomaly_metadata.json")

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        metadata = {
            **self.training_stats,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n  [OK] Model saved to {output_dir}")
        return model_path

    def load_model(self, model_dir: str = "data/models") -> bool:
        """Load a previously trained model."""
        model_path = os.path.join(model_dir, "anomaly_detector.joblib")
        scaler_path = os.path.join(model_dir, "anomaly_scaler.joblib")
        meta_path = os.path.join(model_dir, "anomaly_metadata.json")

        if not all(os.path.exists(p) for p in [model_path, scaler_path, meta_path]):
            print(f"  [!] Model files not found in {model_dir}")
            return False

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(meta_path, "r") as f:
            self.training_stats = json.load(f)

        self.feature_names = self.training_stats.get("feature_names", [])
        self.is_fitted = True
        print(f"  [OK] Model loaded from {model_dir}")
        return True
