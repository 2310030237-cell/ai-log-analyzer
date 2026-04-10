"""
AI-DDE Log Analyzer Dashboard - Streamlit App
Minimal, clean design.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# -- Palette --
# Background: #0e1117 (Streamlit dark default)
# Surface:    #161b22
# Border:     #21262d
# Muted text: #7d8590
# Text:       #c9d1d9
# Accent:     #58a6ff (soft blue)
# Success:    #3fb950
# Warning:    #d29922
# Danger:     #f85149

ACCENT = "#58a6ff"
SURFACE = "#161b22"
BORDER = "#21262d"
MUTED = "#7d8590"
TEXT = "#c9d1d9"
SUCCESS = "#3fb950"
WARNING = "#d29922"
DANGER = "#f85149"

# Minimal chart palette - muted, harmonious
CHART_COLORS = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff"]
CHART_MONO = "#58a6ff"

st.set_page_config(
    page_title="Log Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    .stApp {{ font-family: 'Inter', -apple-system, sans-serif; }}

    /* Metric cards - flat, borderless */
    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 1rem 1.25rem;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: {TEXT} !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {SURFACE};
        border-right: 1px solid {BORDER};
    }}

    /* Sidebar nav radio - force all text white */
    [data-testid="stSidebar"] [role="radiogroup"] label p,
    [data-testid="stSidebar"] [role="radiogroup"] label span,
    [data-testid="stSidebar"] [role="radiogroup"] label div,
    [data-testid="stSidebar"] [role="radiogroup"] label {{
        color: #e6edf3 !important;
        font-size: 0.9rem !important;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover p,
    [data-testid="stSidebar"] [role="radiogroup"] label:hover span,
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        color: {ACCENT} !important;
    }}
    /* Active/checked radio */
    [data-testid="stSidebar"] [role="radio"][aria-checked="true"] p,
    [data-testid="stSidebar"] [role="radio"][aria-checked="true"] span,
    [data-testid="stSidebar"] [role="radio"][aria-checked="true"] div {{
        color: {ACCENT} !important;
        font-weight: 600 !important;
    }}

    /* Headers */
    h1 {{ color: {TEXT} !important; font-weight: 600 !important; font-size: 1.5rem !important; }}
    h2 {{ color: {TEXT} !important; font-weight: 500 !important; font-size: 1.2rem !important; }}
    h3 {{ color: {MUTED} !important; font-weight: 500 !important; font-size: 1rem !important; }}

    /* Page header */
    .page-header {{
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid {BORDER};
        margin-bottom: 1.5rem;
    }}
    .page-header h1 {{
        font-size: 1.4rem !important;
        margin: 0 !important;
        color: {TEXT} !important;
    }}
    .page-header p {{
        color: {MUTED};
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }}

    /* Hide Streamlit chrome */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}

    /* Expander */
    .streamlit-expanderHeader {{
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }}

    /* Divider */
    hr {{ border-color: {BORDER} !important; opacity: 0.5; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 0;
        padding: 10px 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }}

    /* Data frame */
    [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)


def _chart_layout(title="", height=340):
    """Shared minimal chart layout."""
    return dict(
        title=dict(text=title, font=dict(size=13, color=MUTED), x=0, xanchor="left"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color=MUTED),
        height=height,
        margin=dict(t=36, b=32, l=40, r=16),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        showlegend=True,
        legend=dict(font=dict(size=10)),
    )


def load_data():
    """Load all available data for the dashboard."""
    data = {}
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    reports_dir = os.path.join(PROJECT_ROOT, "data", "reports")
    models_dir = os.path.join(PROJECT_ROOT, "data", "models")

    if os.path.exists(processed_dir):
        csv_files = sorted([f for f in os.listdir(processed_dir) if f.startswith("processed_") and f.endswith(".csv")])
        if csv_files:
            data["processed_df"] = pd.read_csv(os.path.join(processed_dir, csv_files[-1]))
            if "timestamp" in data["processed_df"].columns:
                data["processed_df"]["timestamp"] = pd.to_datetime(data["processed_df"]["timestamp"], errors="coerce")

    agg_path = os.path.join(processed_dir, "aggregations.json")
    if os.path.exists(agg_path):
        with open(agg_path) as f:
            data["aggregations"] = json.load(f)

    anomaly_path = os.path.join(processed_dir, "anomaly_results.csv")
    if os.path.exists(anomaly_path):
        data["anomaly_df"] = pd.read_csv(anomaly_path)

    pattern_path = os.path.join(models_dir, "pattern_info.json")
    if os.path.exists(pattern_path):
        with open(pattern_path) as f:
            data["pattern_info"] = json.load(f)

    nlp_path = os.path.join(reports_dir, "nlp_analysis.json")
    if os.path.exists(nlp_path):
        with open(nlp_path) as f:
            data["nlp_results"] = json.load(f)

    meta_path = os.path.join(models_dir, "anomaly_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            data["anomaly_metadata"] = json.load(f)

    if os.path.exists(reports_dir):
        data["report_files"] = [f for f in os.listdir(reports_dir) if f.endswith(".html")]

    return data


def render_sidebar():
    """Minimal sidebar navigation."""
    with st.sidebar:
        st.markdown(f"<p style='color:{MUTED}; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.25rem;'>AI-DDE</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{TEXT}; font-size:1rem; font-weight:600; margin-top:0;'>Log Analyzer</p>", unsafe_allow_html=True)
        st.divider()

        page = st.radio(
            "Nav",
            ["Overview", "Upload", "Results", "Analytics", "Reports"],
            label_visibility="collapsed",
        )

        st.divider()

        # System status - minimal
        data_dir = os.path.join(PROJECT_ROOT, "data")
        raw_count = len(os.listdir(os.path.join(data_dir, "raw"))) if os.path.exists(os.path.join(data_dir, "raw")) else 0
        processed_exists = os.path.exists(os.path.join(data_dir, "processed"))
        model_exists = os.path.exists(os.path.join(data_dir, "models", "anomaly_detector.joblib"))

        st.markdown(f"<p style='color:#58a6ff; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;'>Status</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#e6edf3; font-size:0.85rem; margin:0.3rem 0;'>Raw files: <strong>{raw_count}</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#e6edf3; font-size:0.85rem; margin:0.3rem 0;'>Processed: <strong style='color:{SUCCESS if processed_exists else DANGER}'>{'Yes' if processed_exists else 'No'}</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#e6edf3; font-size:0.85rem; margin:0.3rem 0;'>Model: <strong style='color:{SUCCESS if model_exists else DANGER}'>{'Trained' if model_exists else 'None'}</strong></p>", unsafe_allow_html=True)

        st.divider()
        st.markdown(f"<p style='color:#484f58; font-size:0.75rem;'>v1.0</p>", unsafe_allow_html=True)

    return page


def page_overview(data):
    """Clean overview page."""
    st.markdown("""
    <div class="page-header">
        <h1>Overview</h1>
        <p>Batch processing summary and key metrics</p>
    </div>
    """, unsafe_allow_html=True)

    if "aggregations" not in data:
        st.info("No processed data yet. Go to **Upload** to get started, or run `python main.py run-all`.")
        return

    aggs = data["aggregations"]
    import plotly.graph_objects as go

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", f"{aggs.get('total_records', 0):,}")
    with col2:
        st.metric("Error rate", f"{aggs.get('error_rate', 0)}%")
    with col3:
        if "anomaly_df" in data:
            st.metric("Anomalies", f"{data['anomaly_df']['is_anomaly'].sum():,}")
        else:
            st.metric("Anomalies", "-")
    with col4:
        if "pattern_info" in data:
            st.metric("Patterns", len(data["pattern_info"].get("cluster_info", {})))
        else:
            st.metric("Patterns", "-")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts - two column
    col1, col2 = st.columns(2)

    with col1:
        if "level_distribution" in aggs:
            levels = aggs["level_distribution"]
            fig = go.Figure(data=[go.Pie(
                labels=list(levels.keys()),
                values=list(levels.values()),
                hole=0.55,
                marker=dict(colors=CHART_COLORS[:len(levels)], line=dict(width=0)),
                textinfo="label+percent",
                textfont=dict(size=11, color=TEXT),
                hoverinfo="label+value",
            )])
            layout = _chart_layout("Log levels", 320)
            layout["showlegend"] = False
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "hourly_distribution" in aggs:
            hours = aggs["hourly_distribution"]
            fig = go.Figure(data=[go.Bar(
                x=list(hours.keys()),
                y=list(hours.values()),
                marker_color=CHART_MONO,
                marker_line_width=0,
                opacity=0.7,
            )])
            layout = _chart_layout("Hourly volume", 320)
            layout["showlegend"] = False
            layout["bargap"] = 0.15
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

    # Daily trend - full width
    if "daily_counts" in aggs:
        daily = aggs["daily_counts"]
        fig = go.Figure(data=[go.Scatter(
            x=list(daily.keys()),
            y=list(daily.values()),
            mode="lines",
            line=dict(color=CHART_MONO, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.06)",
        )])
        layout = _chart_layout("Daily trend", 250)
        layout["showlegend"] = False
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)


def page_upload(data):
    """Upload page."""
    st.markdown("""
    <div class="page-header">
        <h1>Upload & Process</h1>
        <p>Add log files and trigger the batch pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Upload log files",
            type=["log", "txt", "json", "csv"],
            accept_multiple_files=True,
            help="Apache, syslog, JSON, and application log formats"
        )
        if uploaded_files:
            raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
            os.makedirs(raw_dir, exist_ok=True)
            for f in uploaded_files:
                path = os.path.join(raw_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                st.success(f"Saved: {f.name}")

    with col2:
        st.markdown("### Actions")
        if st.button("Run full pipeline", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                import yaml
                config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                from main import cmd_run_all
                cmd_run_all(config)
            st.success("Pipeline complete.")
            st.rerun()

        if st.button("Generate sample data", use_container_width=True):
            with st.spinner("Generating..."):
                import yaml
                config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                from main import cmd_generate
                cmd_generate(config)
            st.success("Sample data generated.")
            st.rerun()

    # Raw files listing
    st.divider()
    st.markdown("### Raw files")
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    if os.path.exists(raw_dir):
        files = [f for f in os.listdir(raw_dir) if not f.startswith("_")]
        if files:
            for f in sorted(files):
                size = os.path.getsize(os.path.join(raw_dir, f))
                st.caption(f"{f}  -  {size/1024:.0f} KB")
        else:
            st.caption("No files yet.")
    else:
        st.caption("No data directory.")


def page_results(data):
    """Results viewer."""
    st.markdown("""
    <div class="page-header">
        <h1>Results</h1>
        <p>Processed logs, anomalies, and discovered patterns</p>
    </div>
    """, unsafe_allow_html=True)

    if "processed_df" not in data:
        st.info("No processed data. Run the pipeline first.")
        return

    df = data["processed_df"]

    # Filters row
    col1, col2, col3 = st.columns(3)
    with col1:
        levels = ["All"] + sorted(df["level"].dropna().unique().tolist()) if "level" in df.columns else ["All"]
        selected_level = st.selectbox("Level", levels)
    with col2:
        services = ["All"] + sorted(df["service"].dropna().unique().tolist()) if "service" in df.columns else ["All"]
        selected_service = st.selectbox("Service", services)
    with col3:
        dates = ["All"] + sorted(df["date"].dropna().unique().tolist()) if "date" in df.columns else ["All"]
        selected_date = st.selectbox("Date", dates)

    filtered = df.copy()
    if "level" in df.columns and selected_level != "All":
        filtered = filtered[filtered["level"] == selected_level]
    if "service" in df.columns and selected_service != "All":
        filtered = filtered[filtered["service"] == selected_service]
    if "date" in df.columns and selected_date != "All":
        filtered = filtered[filtered["date"] == selected_date]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    display_cols = [c for c in filtered.columns if not c.startswith("_")]
    st.dataframe(filtered[display_cols].head(500), use_container_width=True, height=380)

    # Anomalies section
    if "anomaly_df" in data:
        st.divider()
        st.markdown("### Anomalies")
        anom_df = data["anomaly_df"]
        anomalies = anom_df[anom_df["is_anomaly"] == 1]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", f"{len(anomalies):,}")
        with col2:
            st.metric("Rate", f"{len(anomalies)/max(len(anom_df),1)*100:.1f}%")
        with col3:
            if "anomaly_severity" in anomalies.columns:
                st.metric("Critical", len(anomalies[anomalies["anomaly_severity"] == "critical"]))

        if len(anomalies) > 0:
            st.dataframe(anomalies.head(200), use_container_width=True, height=280)

    # Patterns
    if "pattern_info" in data:
        st.divider()
        st.markdown("### Patterns")
        patterns = data["pattern_info"].get("cluster_info", {})
        for pid, info in list(patterns.items())[:10]:
            with st.expander(f"Pattern {pid}  -  {info.get('size', 0):,} logs ({info.get('percentage', 0)}%)"):
                st.caption(f"Keywords: {', '.join(t[0] for t in info.get('top_terms', [])[:8])}")
                st.code(info.get("representative", "")[:200], language=None)


def page_analytics(data):
    """Analytics with charts."""
    st.markdown("""
    <div class="page-header">
        <h1>Analytics</h1>
        <p>Visual analysis of logs, anomalies, and NLP insights</p>
    </div>
    """, unsafe_allow_html=True)

    if "aggregations" not in data:
        st.info("No data available. Run the pipeline first.")
        return

    import plotly.graph_objects as go
    aggs = data["aggregations"]

    tab1, tab2, tab3 = st.tabs(["Logs", "Anomalies", "NLP"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            if "status_distribution" in aggs:
                statuses = aggs["status_distribution"]
                colors = []
                for s in statuses.keys():
                    code = int(float(s))
                    if code < 300: colors.append(SUCCESS)
                    elif code < 400: colors.append(WARNING)
                    elif code < 500: colors.append(DANGER)
                    else: colors.append("#f0514a")

                fig = go.Figure(data=[go.Bar(
                    x=list(statuses.keys()),
                    y=list(statuses.values()),
                    marker_color=colors, marker_line_width=0, opacity=0.8,
                )])
                layout = _chart_layout("Status codes", 320)
                layout["showlegend"] = False
                layout["bargap"] = 0.2
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "top_services" in aggs:
                services = aggs["top_services"]
                fig = go.Figure(data=[go.Bar(
                    x=list(services.values()),
                    y=list(services.keys()),
                    orientation="h",
                    marker_color=CHART_MONO, marker_line_width=0, opacity=0.7,
                )])
                layout = _chart_layout("By service", 320)
                layout["showlegend"] = False
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

        if "time_bucket_distribution" in aggs:
            buckets = aggs["time_bucket_distribution"]
            fig = go.Figure(data=[go.Bar(
                x=list(buckets.keys()),
                y=list(buckets.values()),
                marker_color=[CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(buckets))],
                marker_line_width=0, opacity=0.7,
            )])
            layout = _chart_layout("Time of day", 260)
            layout["showlegend"] = False
            layout["bargap"] = 0.3
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

        if "duration_stats" in aggs:
            st.markdown("### Response duration")
            dur = aggs["duration_stats"]
            cols = st.columns(5)
            for c, (label, key) in zip(cols, [("Mean", "mean"), ("P50", "p50"), ("P95", "p95"), ("P99", "p99"), ("Max", "max")]):
                with c:
                    v = dur.get(key)
                    st.metric(label, f"{v:.0f}ms" if v else "-")

    with tab2:
        if "anomaly_df" in data:
            anom_df = data["anomaly_df"]

            fig = go.Figure(data=[go.Histogram(
                x=anom_df["anomaly_score"],
                nbinsx=50,
                marker_color=CHART_MONO, marker_line_width=0, opacity=0.7,
            )])
            layout = _chart_layout("Score distribution", 360)
            layout["showlegend"] = False
            layout["xaxis"]["title"] = dict(text="Score (lower = more anomalous)", font=dict(size=10, color=MUTED))
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

            if "anomaly_severity" in anom_df.columns:
                sev_counts = anom_df["anomaly_severity"].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=sev_counts.index.tolist(),
                    values=sev_counts.values.tolist(),
                    hole=0.55,
                    marker=dict(colors=[DANGER, WARNING, CHART_MONO, SUCCESS], line=dict(width=0)),
                    textinfo="label+percent",
                    textfont=dict(size=11, color=TEXT),
                )])
                layout = _chart_layout("Severity", 300)
                layout["showlegend"] = False
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

            if "anomaly_metadata" in data:
                st.markdown("### Model details")
                st.json(data["anomaly_metadata"])
        else:
            st.info("No anomaly data. Run the pipeline.")

    with tab3:
        if "nlp_results" in data:
            nlp = data["nlp_results"]

            if "severity_analysis" in nlp:
                sev = nlp["severity_analysis"].get("severity_counts", {})
                fig = go.Figure(data=[go.Bar(
                    x=list(sev.keys()),
                    y=list(sev.values()),
                    marker_color=[DANGER, WARNING, CHART_MONO, SUCCESS],
                    marker_line_width=0, opacity=0.8,
                )])
                layout = _chart_layout("Severity keywords", 300)
                layout["showlegend"] = False
                layout["bargap"] = 0.3
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

            if "keywords" in nlp:
                kws = nlp["keywords"][:15]
                fig = go.Figure(data=[go.Bar(
                    x=[k["count"] for k in kws],
                    y=[k["keyword"] for k in kws],
                    orientation="h",
                    marker_color=CHART_MONO, marker_line_width=0, opacity=0.7,
                )])
                layout = _chart_layout("Top keywords", 400)
                layout["showlegend"] = False
                layout["yaxis"]["autorange"] = "reversed"
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

            wc_path = os.path.join(PROJECT_ROOT, "data", "reports", "wordcloud.png")
            if os.path.exists(wc_path):
                st.markdown("### Word cloud")
                st.image(wc_path, use_container_width=True)

            if "summary" in nlp:
                st.markdown("### Insights")
                for insight in nlp["summary"].get("insights", []):
                    st.caption(f"- {insight}")
        else:
            st.info("No NLP data. Run the pipeline.")


def page_reports(data):
    """Report download page."""
    st.markdown("""
    <div class="page-header">
        <h1>Reports</h1>
        <p>Download generated analysis reports</p>
    </div>
    """, unsafe_allow_html=True)

    reports_dir = os.path.join(PROJECT_ROOT, "data", "reports")
    if not os.path.exists(reports_dir):
        st.info("No reports yet. Run the pipeline.")
        return

    report_files = data.get("report_files", [])
    if not report_files:
        st.info("No reports found. Run `python main.py run-all`.")
        return

    for rf in sorted(report_files, reverse=True):
        filepath = os.path.join(reports_dir, rf)
        size = os.path.getsize(filepath)
        mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.markdown(f"**{rf}**")
            st.caption(f"{mtime.strftime('%Y-%m-%d %H:%M')}  |  {size/1024:.0f} KB")
        with col2:
            with open(filepath, "r", encoding="utf-8") as f:
                st.download_button("Download", data=f.read(), file_name=rf, mime="text/html", key=f"dl_{rf}")
        with col3:
            if st.button("Preview", key=f"prev_{rf}"):
                with open(filepath, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=700, scrolling=True)

    json_files = [f for f in os.listdir(reports_dir) if f.endswith(".json") and "report" in f]
    if json_files:
        st.divider()
        st.markdown("### JSON")
        for jf in sorted(json_files, reverse=True):
            filepath = os.path.join(reports_dir, jf)
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(jf)
            with col2:
                with open(filepath, "r") as f:
                    st.download_button("Download", data=f.read(), file_name=jf, mime="application/json", key=f"dl_{jf}")


def main():
    page = render_sidebar()
    data = load_data()

    pages = {
        "Overview": page_overview,
        "Upload": page_upload,
        "Results": page_results,
        "Analytics": page_analytics,
        "Reports": page_reports,
    }
    pages[page](data)


if __name__ == "__main__":
    main()
