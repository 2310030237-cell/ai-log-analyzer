# AI-Based Log File Analyzer with Batch Processing System

A production-ready batch processing AI system that analyzes large-scale log data efficiently, providing meaningful insights for monitoring, security, and decision-making.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Sources   │────▶│  Storage Layer   │────▶│ Batch Processing│
│  (Log Files)    │     │  (Data Lake)     │     │  (PySpark)      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐     ┌────────▼────────┐
                        │   Dashboard     │◀────│  AI/ML Layer    │
                        │  (Streamlit)    │     │ (scikit-learn)  │
                        └─────────────────┘     └────────┬────────┘
                                                         │
                                                ┌────────▼────────┐
                                                │  Reports &      │
                                                │  Insights       │
                                                └─────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample log data
python main.py generate

# 3. Run the full pipeline
python main.py run-all

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py generate` | Generate synthetic log data |
| `python main.py process` | Run batch processing (clean + transform) |
| `python main.py analyze` | Run AI/ML analysis |
| `python main.py report` | Generate analytical reports |
| `python main.py run-all` | Execute full pipeline |
| `python main.py schedule` | Start batch scheduler |

## Project Structure

```
├── config/config.yaml          # Central configuration
├── data/                       # Data lake (raw, processed, models, reports)
├── src/
│   ├── data_generator/         # Synthetic log generation
│   ├── storage/                # HDFS-compatible storage abstraction
│   ├── processing/             # Spark batch processing engine
│   ├── ml/                     # AI/ML models
│   ├── reporting/              # Report generation
│   └── scheduler/              # Batch job scheduling
├── dashboard/                  # Streamlit web dashboard
├── tests/                      # Unit tests
├── main.py                     # CLI entry point
└── requirements.txt
```

## Tech Stack

- **Processing**: Apache Spark (PySpark)
- **ML**: scikit-learn (Isolation Forest, K-Means, DBSCAN)
- **NLP**: NLTK
- **Visualization**: Plotly, Matplotlib
- **Dashboard**: Streamlit
- **Storage**: Local filesystem (HDFS-compatible abstraction)
