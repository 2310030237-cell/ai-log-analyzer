"""
AI-DDE Log Analyzer - Main CLI Entry Point
Orchestrates the full batch processing pipeline.

Usage:
    python main.py generate     Generate synthetic log data
    python main.py process      Run batch processing
    python main.py analyze      Run AI/ML analysis
    python main.py report       Generate reports
    python main.py run-all      Execute full pipeline
    python main.py schedule     Start batch scheduler
"""

import os
import sys
import json
import argparse
import yaml
import pandas as pd
from datetime import datetime


# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def cmd_generate(config: dict):
    """Generate synthetic log data."""
    from src.data_generator.log_generator import generate_sample_data

    gen_config = config.get("generator", {})
    gen_config["output_dir"] = config.get("paths", {}).get("raw_logs", "data/raw")
    generate_sample_data(gen_config)


def cmd_process(config: dict) -> pd.DataFrame:
    """Run batch processing on raw log files."""
    from src.processing.log_parser import LogParser
    from src.processing.spark_processor import BatchProcessor

    raw_dir = config.get("paths", {}).get("raw_logs", "data/raw")
    processed_dir = config.get("paths", {}).get("processed", "data/processed")

    # Parse all raw log files
    parser = LogParser()
    all_records = []

    print("\n" + "=" * 60)
    print("LOG PARSING")
    print("=" * 60)

    log_files = [f for f in os.listdir(raw_dir)
                 if os.path.isfile(os.path.join(raw_dir, f)) and not f.startswith("_")]

    if not log_files:
        print(f"  [!] No log files found in {raw_dir}")
        print(f"  Run 'python main.py generate' first.")
        return pd.DataFrame()

    for filename in sorted(log_files):
        filepath = os.path.join(raw_dir, filename)
        print(f"\n  Parsing: {filename}")
        records = parser.parse_file(filepath)
        all_records.extend(records)
        stats = parser.get_stats()
        print(f"    Records: {len(records):,} | Errors: {stats['parse_errors']} | Success: {stats['success_rate']:.1f}%")
        parser.parse_errors = 0
        parser.total_parsed = 0

    print(f"\n  Total parsed records: {len(all_records):,}")

    # Batch process
    processor = BatchProcessor(config)
    df = processor.process(all_records, processed_dir)
    processor.stop()
    return df


def cmd_analyze(config: dict, df: pd.DataFrame = None) -> dict:
    """Run AI/ML analysis."""
    from src.processing.feature_extractor import FeatureExtractor
    from src.ml.anomaly_detector import AnomalyDetector
    from src.ml.pattern_recognizer import PatternRecognizer
    from src.ml.nlp_analyzer import NLPAnalyzer

    processed_dir = config.get("paths", {}).get("processed", "data/processed")
    models_dir = config.get("paths", {}).get("models", "data/models")
    reports_dir = config.get("paths", {}).get("reports", "data/reports")

    # Load processed data if not provided
    if df is None:
        csv_files = [f for f in os.listdir(processed_dir) if f.endswith(".csv")]
        if not csv_files:
            print("  [!] No processed data found. Run 'python main.py process' first.")
            return {}
        latest = sorted(csv_files)[-1]
        df = pd.read_csv(os.path.join(processed_dir, latest))
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        print(f"  Loaded {len(df):,} records from {latest}")

    results = {}

    # Feature extraction
    extractor = FeatureExtractor(config)
    features = extractor.extract_features(df, mode="record")
    print(f"\n  Extracted {features.shape[1]} features for {features.shape[0]:,} records")

    # 1. Anomaly Detection
    detector = AnomalyDetector(config)
    detector.train(features)
    anomaly_results = detector.predict(features)
    anomaly_summary = detector.get_anomaly_summary(anomaly_results)
    detector.save_model(models_dir)
    results["anomaly_summary"] = anomaly_summary
    results["anomaly_results"] = anomaly_results

    # Save anomaly results
    anomaly_out = anomaly_results[["is_anomaly", "anomaly_score", "anomaly_severity"]].copy()
    if "timestamp" in df.columns:
        anomaly_out["timestamp"] = df["timestamp"].values[:len(anomaly_out)]
    if "message" in df.columns:
        anomaly_out["message"] = df["message"].values[:len(anomaly_out)]
    anomaly_out.to_csv(os.path.join(processed_dir, "anomaly_results.csv"), index=False)

    # 2. Pattern Recognition
    if "message" in df.columns:
        recognizer = PatternRecognizer(config)
        pattern_results = recognizer.fit_predict(df["message"])
        pattern_summary = recognizer.get_pattern_summary()
        recognizer.save_model(models_dir)
        results["pattern_summary"] = pattern_summary
        results["pattern_results"] = pattern_results

    # 3. NLP Analysis
    if "message" in df.columns:
        nlp = NLPAnalyzer(config)
        nlp_results = nlp.analyze(df["message"])
        nlp.generate_wordcloud(df["message"], os.path.join(reports_dir, "wordcloud.png"))
        nlp.save_results(nlp_results, reports_dir)
        results["nlp_results"] = nlp_results

    return results


def cmd_report(config: dict, df: pd.DataFrame = None, analysis_results: dict = None):
    """Generate analytical reports."""
    from src.reporting.report_generator import ReportGenerator

    processed_dir = config.get("paths", {}).get("processed", "data/processed")

    # Load data if needed
    if df is None:
        csv_files = [f for f in os.listdir(processed_dir) if f.startswith("processed_") and f.endswith(".csv")]
        if not csv_files:
            print("  [!] No processed data. Run pipeline first.")
            return
        latest = sorted(csv_files)[-1]
        df = pd.read_csv(os.path.join(processed_dir, latest))

    # Load aggregations
    agg_path = os.path.join(processed_dir, "aggregations.json")
    if os.path.exists(agg_path):
        with open(agg_path, "r") as f:
            aggregations = json.load(f)
    else:
        aggregations = {"total_records": len(df)}

    # Extract analysis components
    anomaly_summary = analysis_results.get("anomaly_summary") if analysis_results else None
    pattern_summary = analysis_results.get("pattern_summary") if analysis_results else None
    nlp_results = analysis_results.get("nlp_results") if analysis_results else None

    # Generate report
    generator = ReportGenerator(config)
    report_path = generator.generate_full_report(
        processed_df=df,
        aggregations=aggregations,
        anomaly_summary=anomaly_summary,
        pattern_summary=pattern_summary,
        nlp_results=nlp_results,
    )
    return report_path


def cmd_run_all(config: dict):
    """Execute the full pipeline: generate -> process -> analyze -> report."""
    print("\n" + "#" * 60)
    print("#  AI-DDE LOG ANALYZER - FULL PIPELINE")
    print("#" * 60)
    start = datetime.now()

    # Step 1: Generate (only if no raw data exists)
    raw_dir = config.get("paths", {}).get("raw_logs", "data/raw")
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        print("\n>> STEP 1: Generating sample data...")
        cmd_generate(config)
    else:
        print(f"\n>> STEP 1: Using existing data in {raw_dir}")

    # Step 2: Process
    print("\n>> STEP 2: Batch processing...")
    df = cmd_process(config)

    if df.empty:
        print("  Pipeline aborted: no data to process.")
        return

    # Step 3: Analyze
    print("\n>> STEP 3: AI/ML analysis...")
    analysis_results = cmd_analyze(config, df)

    # Step 4: Report
    print("\n>> STEP 4: Generating reports...")
    cmd_report(config, df, analysis_results)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'#'*60}")
    print(f"#  PIPELINE COMPLETE - {elapsed:.1f}s elapsed")
    print(f"{'#'*60}")


def cmd_schedule(config: dict):
    """Start the batch scheduler."""
    from src.scheduler.batch_scheduler import BatchScheduler

    scheduler = BatchScheduler(config)
    scheduler.set_job(lambda: cmd_run_all(config))
    scheduler.start()


def main():
    parser = argparse.ArgumentParser(
        description="AI-DDE Log Analyzer - Batch Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  generate   Generate synthetic log data for testing
  process    Run batch processing (parse, clean, transform)
  analyze    Run AI/ML analysis (anomaly detection, patterns, NLP)
  report     Generate analytical reports
  run-all    Execute the full pipeline end-to-end
  schedule   Start the batch scheduler for periodic execution
        """,
    )
    parser.add_argument("command", choices=["generate", "process", "analyze", "report", "run-all", "schedule"],
                        help="Pipeline command to execute")
    parser.add_argument("--config", default=None, help="Path to config YAML file")

    args = parser.parse_args()
    config = load_config(args.config)

    commands = {
        "generate": cmd_generate,
        "process": cmd_process,
        "analyze": cmd_analyze,
        "report": cmd_report,
        "run-all": cmd_run_all,
        "schedule": cmd_schedule,
    }

    commands[args.command](config)


if __name__ == "__main__":
    main()
