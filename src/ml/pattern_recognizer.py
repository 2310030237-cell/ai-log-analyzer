"""
Pattern Recognition Module
Uses clustering (K-Means / DBSCAN) and TF-IDF to identify
recurring patterns and group similar log events.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


class PatternRecognizer:
    """
    Identifies recurring patterns in log messages using
    TF-IDF vectorization and clustering algorithms.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        ml_config = self.config.get("ml", {}).get("pattern_recognition", {})

        self.algorithm = ml_config.get("algorithm", "kmeans")
        self.n_clusters = ml_config.get("n_clusters", 10)
        self.max_features = ml_config.get("max_features", 5000)

        self.vectorizer = None
        self.svd = None
        self.model = None
        self.is_fitted = False
        self.cluster_info = {}

    def fit_predict(self, messages: pd.Series) -> pd.DataFrame:
        """
        Vectorize log messages and cluster them into patterns.

        Args:
            messages: Series of log message strings

        Returns:
            DataFrame with cluster assignments and pattern info
        """
        print("\n" + "=" * 60)
        print("PATTERN RECOGNITION")
        print("=" * 60)

        # Clean messages
        clean_msgs = messages.fillna("").astype(str)
        clean_msgs = clean_msgs[clean_msgs.str.len() > 0]

        print(f"  Messages: {len(clean_msgs):,}")
        print(f"  Algorithm: {self.algorithm.upper()}")

        # TF-IDF Vectorization
        print("\n  [1/3] TF-IDF Vectorization...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]*\b',
        )
        tfidf_matrix = self.vectorizer.fit_transform(clean_msgs)
        print(f"    Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        print(f"    Matrix shape: {tfidf_matrix.shape}")

        # Dimensionality reduction for better clustering
        n_components = min(50, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        if n_components > 1:
            print(f"\n  [2/3] Dimensionality reduction (SVD -> {n_components})...")
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            X_reduced = self.svd.fit_transform(tfidf_matrix)
            explained_var = self.svd.explained_variance_ratio_.sum()
            print(f"    Explained variance: {explained_var:.2%}")
        else:
            X_reduced = tfidf_matrix.toarray()

        # Clustering
        print(f"\n  [3/3] Clustering ({self.algorithm})...")
        if self.algorithm == "kmeans":
            # Adjust n_clusters if needed
            actual_clusters = min(self.n_clusters, len(clean_msgs) - 1)
            self.model = KMeans(
                n_clusters=actual_clusters,
                random_state=42,
                n_init=10,
                max_iter=300,
            )
            labels = self.model.fit_predict(X_reduced)
        elif self.algorithm == "dbscan":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_reduced)
            self.model = DBSCAN(eps=0.5, min_samples=5)
            labels = self.model.fit_predict(X_scaled)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        self.is_fitted = True

        # Build results
        results = pd.DataFrame({
            "message": clean_msgs.values,
            "cluster": labels,
        })

        # Analyze clusters
        self._analyze_clusters(results, tfidf_matrix)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        print(f"\n  [OK] Found {n_clusters} distinct patterns")
        if n_noise > 0:
            print(f"    Noise points: {n_noise:,}")

        return results

    def _analyze_clusters(self, results: pd.DataFrame, tfidf_matrix):
        """Analyze each cluster to extract representative patterns."""
        feature_names = self.vectorizer.get_feature_names_out()
        self.cluster_info = {}

        for cluster_id in sorted(results["cluster"].unique()):
            if cluster_id == -1:
                continue  # Skip noise cluster in DBSCAN

            mask = results["cluster"] == cluster_id
            cluster_msgs = results[mask]["message"]
            cluster_tfidf = tfidf_matrix[mask.values]

            # Top terms for this cluster
            mean_tfidf = cluster_tfidf.mean(axis=0).A1 if hasattr(cluster_tfidf, 'A1') else cluster_tfidf.mean(axis=0)
            if hasattr(mean_tfidf, 'A1'):
                mean_tfidf = mean_tfidf.A1
            elif hasattr(mean_tfidf, 'toarray'):
                mean_tfidf = np.array(mean_tfidf).flatten()
            else:
                mean_tfidf = np.array(mean_tfidf).flatten()

            top_indices = mean_tfidf.argsort()[-10:][::-1]
            top_terms = [(feature_names[i], round(float(mean_tfidf[i]), 4)) for i in top_indices]

            # Representative message (closest to centroid)
            representative = cluster_msgs.iloc[0] if len(cluster_msgs) > 0 else ""

            # Common words in cluster
            all_words = " ".join(cluster_msgs.values).split()
            common_words = Counter(all_words).most_common(10)

            self.cluster_info[int(cluster_id)] = {
                "size": int(mask.sum()),
                "percentage": round(mask.sum() / len(results) * 100, 2),
                "top_terms": top_terms,
                "representative": representative[:200],
                "common_words": common_words,
            }

    def get_pattern_summary(self) -> Dict:
        """Get a summary of discovered patterns."""
        if not self.cluster_info:
            return {"error": "No patterns analyzed yet. Run fit_predict() first."}

        summary = {
            "n_patterns": len(self.cluster_info),
            "algorithm": self.algorithm,
            "patterns": {},
        }

        for cid, info in self.cluster_info.items():
            summary["patterns"][f"pattern_{cid}"] = {
                "size": info["size"],
                "percentage": info["percentage"],
                "top_keywords": [t[0] for t in info["top_terms"][:5]],
                "representative_message": info["representative"],
            }

        return summary

    def save_model(self, output_dir: str = "data/models") -> str:
        """Save the pattern recognition model."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit_predict() first.")

        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "pattern_recognizer.joblib")
        vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.joblib")
        info_path = os.path.join(output_dir, "pattern_info.json")

        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        if self.svd:
            joblib.dump(self.svd, os.path.join(output_dir, "svd_transformer.joblib"))

        with open(info_path, "w") as f:
            json.dump({
                "algorithm": self.algorithm,
                "n_clusters": self.n_clusters,
                "cluster_info": {str(k): {
                    "size": v["size"],
                    "percentage": v["percentage"],
                    "top_terms": v["top_terms"],
                    "representative": v["representative"],
                } for k, v in self.cluster_info.items()},
                "saved_at": datetime.now().isoformat(),
            }, f, indent=2)

        print(f"  [OK] Pattern model saved to {output_dir}")
        return model_path

    def load_model(self, model_dir: str = "data/models") -> bool:
        """Load a previously saved pattern recognition model."""
        model_path = os.path.join(model_dir, "pattern_recognizer.joblib")
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")

        if not all(os.path.exists(p) for p in [model_path, vectorizer_path]):
            return False

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        svd_path = os.path.join(model_dir, "svd_transformer.joblib")
        if os.path.exists(svd_path):
            self.svd = joblib.load(svd_path)

        info_path = os.path.join(model_dir, "pattern_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                data = json.load(f)
                self.cluster_info = {int(k): v for k, v in data.get("cluster_info", {}).items()}

        self.is_fitted = True
        return True
