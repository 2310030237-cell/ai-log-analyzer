"""
Spark Batch Processing Engine
Handles data cleaning, transformation, and aggregation using PySpark.
Falls back to Pandas if PySpark is not available.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Try PySpark, fall back to Pandas-only mode
SPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, FloatType
    SPARK_AVAILABLE = True
except ImportError:
    pass


class BatchProcessor:
    """
    Batch processing engine for log data.
    Uses PySpark when available, falls back to Pandas.
    """

    def __init__(self, config: dict):
        self.config = config
        self.spark_config = config.get("spark", {})
        self.processing_config = config.get("processing", {})
        self.spark = None
        self.use_spark = SPARK_AVAILABLE

        if self.use_spark:
            self._init_spark()

    def _init_spark(self):
        """Initialize SparkSession."""
        try:
            self.spark = (
                SparkSession.builder
                .appName(self.spark_config.get("app_name", "LogAnalyzer"))
                .master(self.spark_config.get("master", "local[*]"))
                .config("spark.driver.memory", self.spark_config.get("driver_memory", "2g"))
                .config("spark.executor.memory", self.spark_config.get("executor_memory", "2g"))
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.ui.showConsoleProgress", "false")
                .getOrCreate()
            )
            self.spark.sparkContext.setLogLevel(self.spark_config.get("log_level", "WARN"))
            print("  [OK] SparkSession initialized (local mode)")
        except Exception as e:
            print(f"  [!] Spark initialization failed: {e}")
            print("  -> Falling back to Pandas processing")
            self.use_spark = False

    def process(self, records: List[Dict], output_dir: str = "data/processed") -> pd.DataFrame:
        """
        Execute the full batch processing pipeline.

        Pipeline: Raw Records -> Clean -> Transform -> Aggregate -> Save

        Args:
            records: List of parsed log record dicts
            output_dir: Directory for processed output

        Returns:
            Processed Pandas DataFrame
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("BATCH PROCESSING PIPELINE")
        print("=" * 60)
        print(f"  Engine: {'PySpark' if self.use_spark else 'Pandas'}")
        print(f"  Input records: {len(records):,}")

        # Step 1: Convert to DataFrame
        print("\n[1/4] Converting to DataFrame...")
        df = self._to_dataframe(records)
        print(f"  -> {len(df):,} records loaded")

        # Step 2: Clean
        print("\n[2/4] Cleaning data...")
        df = self._clean(df)
        print(f"  -> {len(df):,} records after cleaning")

        # Step 3: Transform
        print("\n[3/4] Transforming data...")
        df = self._transform(df)
        print(f"  -> {len(df):,} records after transformation")

        # Step 4: Save
        print("\n[4/4] Saving processed data...")
        self._save(df, output_dir)
        print(f"  -> Saved to {output_dir}")

        # Generate aggregations
        print("\n[+] Computing aggregations...")
        aggs = self._aggregate(df)
        agg_path = os.path.join(output_dir, "aggregations.json")
        with open(agg_path, "w") as f:
            json.dump(aggs, f, indent=2, default=str)
        print(f"  -> Aggregations saved to {agg_path}")

        print(f"\n[OK] Batch processing complete. {len(df):,} records processed.")
        return df

    def _to_dataframe(self, records: List[Dict]) -> pd.DataFrame:
        """Convert parsed records to a DataFrame."""
        df = pd.DataFrame(records)

        # Ensure timestamp column exists and is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Ensure level column
        if "level" not in df.columns:
            df["level"] = "INFO"

        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data:
        - Remove duplicates
        - Handle nulls
        - Remove invalid entries
        - Filter short messages
        """
        initial_count = len(df)

        # Remove exact duplicates
        if "_raw" in df.columns:
            df = df.drop_duplicates(subset=["_raw"], keep="first")
            dupes_removed = initial_count - len(df)
            print(f"    Duplicates removed: {dupes_removed:,}")
        else:
            df = df.drop_duplicates(keep="first")

        # Handle nulls in critical columns
        null_handling = self.processing_config.get("null_handling", "drop")
        critical_cols = ["timestamp", "message"]
        existing_critical = [c for c in critical_cols if c in df.columns]

        if null_handling == "drop" and existing_critical:
            before = len(df)
            df = df.dropna(subset=existing_critical)
            print(f"    Null rows dropped: {before - len(df):,}")
        elif null_handling == "fill":
            if "message" in df.columns:
                df["message"] = df["message"].fillna("")
            if "level" in df.columns:
                df["level"] = df["level"].fillna("INFO")

        # Remove entries with very short messages
        min_len = self.processing_config.get("min_message_length", 5)
        if "message" in df.columns:
            before = len(df)
            df = df[df["message"].str.len() >= min_len]
            print(f"    Short messages filtered: {before - len(df):,}")

        # Remove invalid timestamps
        if "timestamp" in df.columns:
            before = len(df)
            df = df.dropna(subset=["timestamp"])
            print(f"    Invalid timestamps removed: {before - len(df):,}")

        print(f"    Total cleaned: {initial_count - len(df):,} records removed")
        return df.reset_index(drop=True)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform and enrich the data:
        - Extract time components
        - Normalize log levels
        - Add derived columns
        """
        # Normalize log levels
        level_map = {"WARN": "WARNING", "FATAL": "CRITICAL"}
        if "level" in df.columns:
            df["level"] = df["level"].str.upper().replace(level_map)

        # Extract time components
        if "timestamp" in df.columns:
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["date"] = df["timestamp"].dt.date.astype(str)
            df["is_business_hours"] = df["hour"].between(9, 17)
            df["time_bucket"] = pd.cut(
                df["hour"],
                bins=[0, 6, 12, 18, 24],
                labels=["night", "morning", "afternoon", "evening"],
                include_lowest=True
            )

        # Add severity numeric column
        severity_map = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
        if "level" in df.columns:
            df["severity_num"] = df["level"].map(severity_map).fillna(1).astype(int)

        # Message length feature
        if "message" in df.columns:
            df["message_length"] = df["message"].str.len()

        # Status code category
        if "status_code" in df.columns:
            df["status_category"] = pd.cut(
                df["status_code"],
                bins=[0, 199, 299, 399, 499, 599],
                labels=["1xx", "2xx", "3xx", "4xx", "5xx"]
            )

        return df

    def _aggregate(self, df: pd.DataFrame) -> Dict:
        """Compute aggregations for reporting."""
        aggs = {
            "total_records": len(df),
            "processing_timestamp": datetime.now().isoformat(),
        }

        # Log level distribution
        if "level" in df.columns:
            aggs["level_distribution"] = df["level"].value_counts().to_dict()

        # Hourly distribution
        if "hour" in df.columns:
            aggs["hourly_distribution"] = df["hour"].value_counts().sort_index().to_dict()
            aggs["hourly_distribution"] = {str(k): v for k, v in aggs["hourly_distribution"].items()}

        # Daily counts
        if "date" in df.columns:
            daily = df["date"].value_counts().sort_index()
            aggs["daily_counts"] = {str(k): int(v) for k, v in daily.items()}

        # Error rate
        if "level" in df.columns:
            error_count = len(df[df["level"].isin(["ERROR", "CRITICAL"])])
            aggs["error_count"] = error_count
            aggs["error_rate"] = round(error_count / len(df) * 100, 2) if len(df) > 0 else 0

        # Top endpoints (for Apache logs)
        if "endpoint" in df.columns:
            aggs["top_endpoints"] = df["endpoint"].value_counts().head(10).to_dict()

        # Top IPs
        if "ip" in df.columns:
            aggs["top_ips"] = df["ip"].value_counts().head(10).to_dict()

        # Top services
        if "service" in df.columns:
            aggs["top_services"] = df["service"].value_counts().to_dict()

        # Status code distribution
        if "status_code" in df.columns:
            aggs["status_distribution"] = df["status_code"].value_counts().to_dict()
            aggs["status_distribution"] = {str(k): v for k, v in aggs["status_distribution"].items()}

        # Response size stats
        if "response_size" in df.columns:
            rs = df["response_size"].dropna()
            if len(rs) > 0:
                aggs["response_size"] = {
                    "mean": round(rs.mean(), 2),
                    "median": round(rs.median(), 2),
                    "max": int(rs.max()),
                    "total": int(rs.sum()),
                }

        # Duration stats (for JSON logs)
        if "duration_ms" in df.columns:
            dur = df["duration_ms"].dropna()
            if len(dur) > 0:
                aggs["duration_stats"] = {
                    "mean": round(dur.mean(), 2),
                    "p50": round(dur.quantile(0.5), 2),
                    "p95": round(dur.quantile(0.95), 2),
                    "p99": round(dur.quantile(0.99), 2),
                    "max": round(float(dur.max()), 2),
                }

        # Time bucket distribution
        if "time_bucket" in df.columns:
            aggs["time_bucket_distribution"] = df["time_bucket"].value_counts().to_dict()

        return aggs

    def _save(self, df: pd.DataFrame, output_dir: str):
        """Save processed data to disk."""
        # Save as CSV (without raw column to save space)
        save_cols = [c for c in df.columns if c != "_raw"]
        csv_path = os.path.join(output_dir, f"processed_logs_{datetime.now().strftime('%Y%m%d')}.csv")
        df[save_cols].to_csv(csv_path, index=False)
        size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"    CSV: {csv_path} ({size_mb:.1f} MB)")

        # Save summary stats
        stats = {
            "total_records": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": str(df["timestamp"].min()) if "timestamp" in df.columns else None,
                "end": str(df["timestamp"].max()) if "timestamp" in df.columns else None,
            },
            "processing_time": datetime.now().isoformat(),
        }
        stats_path = os.path.join(output_dir, "processing_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

    def stop(self):
        """Stop the Spark session."""
        if self.spark:
            self.spark.stop()
            print("  SparkSession stopped.")
