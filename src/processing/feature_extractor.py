"""
Feature Extractor
Extracts numerical and categorical features from parsed/processed log data
for use in ML models (anomaly detection, pattern recognition).
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter


class FeatureExtractor:
    """Extracts ML-ready features from processed log DataFrames."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.feature_names = []

    def extract_features(self, df: pd.DataFrame, mode: str = "record") -> pd.DataFrame:
        """
        Extract features from processed log data.

        Args:
            df: Processed log DataFrame
            mode: "record" for per-record features, "window" for time-window features

        Returns:
            DataFrame with extracted features
        """
        if mode == "record":
            return self._extract_record_features(df)
        elif mode == "window":
            return self._extract_window_features(df)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'record' or 'window'.")

    def _extract_record_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract per-record features."""
        features = pd.DataFrame(index=df.index)

        # Severity numeric
        if "severity_num" in df.columns:
            features["severity"] = df["severity_num"]
        elif "level" in df.columns:
            severity_map = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
            features["severity"] = df["level"].map(severity_map).fillna(1)

        # Time-based features
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            features["hour"] = ts.dt.hour
            features["day_of_week"] = ts.dt.dayofweek
            features["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
            features["is_night"] = ((ts.dt.hour < 6) | (ts.dt.hour > 22)).astype(int)
            features["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute

            # Cyclical encoding for hour
            features["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
            features["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)

        # Message features
        if "message" in df.columns:
            features["message_length"] = df["message"].str.len().fillna(0)
            features["word_count"] = df["message"].str.split().str.len().fillna(0)
            features["has_error_keyword"] = df["message"].str.contains(
                r'error|fail|exception|crash|timeout|denied|refused',
                case=False, na=False
            ).astype(int)
            features["has_number"] = df["message"].str.contains(r'\d+', na=False).astype(int)
            features["uppercase_ratio"] = df["message"].apply(
                lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
            )
            features["special_char_count"] = df["message"].str.count(r'[!@#$%^&*()_+=\[\]{};:\'"\\|,.<>?/~`]')

        # Network features (Apache logs)
        if "status_code" in df.columns:
            features["status_code"] = df["status_code"].fillna(200)
            features["is_error_status"] = (df["status_code"] >= 400).astype(int)
            features["is_server_error"] = (df["status_code"] >= 500).astype(int)

        if "response_size" in df.columns:
            features["response_size"] = df["response_size"].fillna(0)
            features["response_size_log"] = np.log1p(df["response_size"].fillna(0))

        # Duration features (JSON logs)
        if "duration_ms" in df.columns:
            features["duration_ms"] = df["duration_ms"].fillna(0)
            features["duration_log"] = np.log1p(df["duration_ms"].fillna(0))
            features["is_slow"] = (df["duration_ms"] > 1000).astype(int)

        # Service encoding (if present)
        if "service" in df.columns:
            service_dummies = pd.get_dummies(df["service"], prefix="service")
            features = pd.concat([features, service_dummies], axis=1)

        # Method encoding (Apache)
        if "method" in df.columns:
            method_dummies = pd.get_dummies(df["method"], prefix="method")
            features = pd.concat([features, method_dummies], axis=1)

        # Fill remaining NaN
        features = features.fillna(0)
        self.feature_names = list(features.columns)
        return features

    def _extract_window_features(self, df: pd.DataFrame, window_minutes: int = 60) -> pd.DataFrame:
        """
        Extract time-window aggregated features.
        Groups logs by time windows and computes aggregate statistics.
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column for window features.")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")

        # Create time windows
        df["window"] = df["timestamp"].dt.floor(f"{window_minutes}min")

        windows = []
        for window_start, group in df.groupby("window"):
            features = {
                "window_start": window_start,
                "log_count": len(group),
            }

            # Level distribution
            if "level" in group.columns:
                level_counts = group["level"].value_counts()
                for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                    features[f"count_{level.lower()}"] = level_counts.get(level, 0)
                features["error_rate"] = (
                    (level_counts.get("ERROR", 0) + level_counts.get("CRITICAL", 0)) / len(group)
                )

            # Message stats
            if "message" in group.columns:
                msg_lens = group["message"].str.len()
                features["avg_message_length"] = msg_lens.mean()
                features["max_message_length"] = msg_lens.max()
                features["unique_messages"] = group["message"].nunique()
                features["message_diversity"] = group["message"].nunique() / max(len(group), 1)

            # Unique IPs
            if "ip" in group.columns:
                features["unique_ips"] = group["ip"].nunique()

            # Unique services
            if "service" in group.columns:
                features["unique_services"] = group["service"].nunique()

            # Status code distribution
            if "status_code" in group.columns:
                features["avg_status"] = group["status_code"].mean()
                features["error_status_count"] = (group["status_code"] >= 400).sum()
                features["server_error_count"] = (group["status_code"] >= 500).sum()

            # Response size
            if "response_size" in group.columns:
                features["avg_response_size"] = group["response_size"].mean()
                features["total_response_size"] = group["response_size"].sum()

            # Duration
            if "duration_ms" in group.columns:
                features["avg_duration"] = group["duration_ms"].mean()
                features["p95_duration"] = group["duration_ms"].quantile(0.95)
                features["max_duration"] = group["duration_ms"].max()

            # Severity stats
            if "severity_num" in group.columns:
                features["avg_severity"] = group["severity_num"].mean()
                features["max_severity"] = group["severity_num"].max()

            windows.append(features)

        result = pd.DataFrame(windows)
        result = result.fillna(0)
        self.feature_names = [c for c in result.columns if c != "window_start"]
        return result

    def get_feature_importance_summary(self, features_df: pd.DataFrame) -> Dict:
        """Compute basic feature statistics for interpretability."""
        summary = {}
        for col in features_df.select_dtypes(include=[np.number]).columns:
            summary[col] = {
                "mean": round(features_df[col].mean(), 4),
                "std": round(features_df[col].std(), 4),
                "min": round(float(features_df[col].min()), 4),
                "max": round(float(features_df[col].max()), 4),
                "non_zero_pct": round((features_df[col] != 0).mean() * 100, 2),
            }
        return summary
