"""
Report Generator
Generates structured analytical reports in HTML and JSON formats
with embedded Plotly charts.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ReportGenerator:
    """Generates comprehensive analytical reports from processed log data."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.output_dir = self.config.get("paths", {}).get("reports", "data/reports")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_full_report(
        self,
        processed_df: pd.DataFrame,
        aggregations: Dict,
        anomaly_summary: Dict = None,
        pattern_summary: Dict = None,
        nlp_results: Dict = None,
    ) -> str:
        """
        Generate a comprehensive HTML report with all analysis results.

        Returns:
            Path to the generated HTML report
        """
        print("\n" + "=" * 60)
        print("REPORT GENERATION")
        print("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"log_analysis_report_{timestamp}"

        # Generate charts
        charts_html = ""
        if PLOTLY_AVAILABLE:
            print("  Generating charts...")
            charts_html = self._generate_charts(processed_df, aggregations, anomaly_summary)

        # Build HTML report
        html = self._build_html_report(
            report_name=report_name,
            aggregations=aggregations,
            anomaly_summary=anomaly_summary,
            pattern_summary=pattern_summary,
            nlp_results=nlp_results,
            charts_html=charts_html,
            df=processed_df,
        )

        # Save HTML report
        html_path = os.path.join(self.output_dir, f"{report_name}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  [OK] HTML report: {html_path}")

        # Save JSON report
        json_report = {
            "report_name": report_name,
            "generated_at": datetime.now().isoformat(),
            "aggregations": aggregations,
            "anomaly_summary": anomaly_summary,
            "pattern_summary": pattern_summary,
            "nlp_summary": nlp_results.get("summary") if nlp_results else None,
        }
        json_path = os.path.join(self.output_dir, f"{report_name}.json")
        with open(json_path, "w") as f:
            json.dump(json_report, f, indent=2, default=str)
        print(f"  [OK] JSON report: {json_path}")

        print(f"\n[OK] Reports generated in {self.output_dir}")
        return html_path

    def _generate_charts(self, df: pd.DataFrame, aggs: Dict, anomaly_summary: Dict = None) -> str:
        """Generate Plotly charts and return as HTML strings."""
        charts = []

        # 1. Log Level Distribution (Pie)
        if "level_distribution" in aggs:
            fig = go.Figure(data=[go.Pie(
                labels=list(aggs["level_distribution"].keys()),
                values=list(aggs["level_distribution"].values()),
                hole=0.4,
                marker=dict(colors=["#00d2ff", "#4caf50", "#ff9800", "#f44336", "#9c27b0"]),
            )])
            fig.update_layout(
                title="Log Level Distribution",
                template="plotly_dark",
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#16213e",
                font=dict(color="#e0e0e0"),
                height=400,
            )
            charts.append(fig.to_html(full_html=False, include_plotlyjs=False))

        # 2. Hourly Distribution (Bar)
        if "hourly_distribution" in aggs:
            hours = list(aggs["hourly_distribution"].keys())
            counts = list(aggs["hourly_distribution"].values())
            fig = go.Figure(data=[go.Bar(
                x=hours, y=counts,
                marker_color="#00d2ff",
                marker_line_color="#0a9cf5",
                marker_line_width=1,
            )])
            fig.update_layout(
                title="Hourly Log Distribution",
                xaxis_title="Hour of Day",
                yaxis_title="Log Count",
                template="plotly_dark",
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#16213e",
                font=dict(color="#e0e0e0"),
                height=400,
            )
            charts.append(fig.to_html(full_html=False, include_plotlyjs=False))

        # 3. Daily Trend (Line)
        if "daily_counts" in aggs:
            dates = list(aggs["daily_counts"].keys())
            counts = list(aggs["daily_counts"].values())
            fig = go.Figure(data=[go.Scatter(
                x=dates, y=counts,
                mode="lines+markers",
                line=dict(color="#4caf50", width=2),
                marker=dict(size=6, color="#4caf50"),
                fill="tozeroy",
                fillcolor="rgba(76, 175, 80, 0.1)",
            )])
            fig.update_layout(
                title="Daily Log Volume Trend",
                xaxis_title="Date",
                yaxis_title="Log Count",
                template="plotly_dark",
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#16213e",
                font=dict(color="#e0e0e0"),
                height=400,
            )
            charts.append(fig.to_html(full_html=False, include_plotlyjs=False))

        # 4. Status Code Distribution (if present)
        if "status_distribution" in aggs:
            statuses = list(aggs["status_distribution"].keys())
            counts = list(aggs["status_distribution"].values())
            colors = []
            for s in statuses:
                code = int(float(s))
                if code < 300:
                    colors.append("#4caf50")
                elif code < 400:
                    colors.append("#ff9800")
                elif code < 500:
                    colors.append("#ff5722")
                else:
                    colors.append("#f44336")
            fig = go.Figure(data=[go.Bar(x=statuses, y=counts, marker_color=colors)])
            fig.update_layout(
                title="HTTP Status Code Distribution",
                xaxis_title="Status Code",
                yaxis_title="Count",
                template="plotly_dark",
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#16213e",
                font=dict(color="#e0e0e0"),
                height=400,
            )
            charts.append(fig.to_html(full_html=False, include_plotlyjs=False))

        # 5. Anomaly Score Distribution (if available)
        if anomaly_summary and anomaly_summary.get("severity_distribution"):
            sev = anomaly_summary["severity_distribution"]
            fig = go.Figure(data=[go.Bar(
                x=list(sev.keys()),
                y=list(sev.values()),
                marker_color=["#f44336", "#ff5722", "#ff9800", "#4caf50"],
            )])
            fig.update_layout(
                title="Anomaly Severity Distribution",
                xaxis_title="Severity",
                yaxis_title="Count",
                template="plotly_dark",
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#16213e",
                font=dict(color="#e0e0e0"),
                height=400,
            )
            charts.append(fig.to_html(full_html=False, include_plotlyjs=False))

        return "\n".join(charts)

    def _build_html_report(self, report_name, aggregations, anomaly_summary,
                           pattern_summary, nlp_results, charts_html, df) -> str:
        """Build the full HTML report document."""
        total = aggregations.get("total_records", 0)
        error_rate = aggregations.get("error_rate", 0)
        error_count = aggregations.get("error_count", 0)

        anomaly_count = 0
        anomaly_rate = 0
        if anomaly_summary:
            anomaly_count = anomaly_summary.get("total_anomalies", 0)
            anomaly_rate = anomaly_summary.get("anomaly_rate", 0)

        pattern_count = 0
        if pattern_summary:
            pattern_count = pattern_summary.get("n_patterns", 0)

        # Build pattern details
        pattern_html = ""
        if pattern_summary and pattern_summary.get("patterns"):
            rows = ""
            for pid, info in pattern_summary["patterns"].items():
                kw = ", ".join(info.get("top_keywords", []))
                rows += f"""
                <tr>
                    <td>{pid}</td>
                    <td>{info.get('size', 0):,}</td>
                    <td>{info.get('percentage', 0)}%</td>
                    <td>{kw}</td>
                    <td class="msg-cell">{info.get('representative_message', '')[:100]}</td>
                </tr>"""
            pattern_html = f"""
            <div class="section">
                <h2>🔍 Discovered Patterns</h2>
                <table>
                    <thead><tr><th>Pattern</th><th>Size</th><th>%</th><th>Keywords</th><th>Example</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>"""

        # NLP insights
        nlp_html = ""
        if nlp_results and nlp_results.get("summary"):
            insights = nlp_results["summary"].get("insights", [])
            insights_li = "".join(f"<li>{i}</li>" for i in insights)
            nlp_html = f"""
            <div class="section">
                <h2>📝 NLP Insights</h2>
                <ul>{insights_li}</ul>
            </div>"""

            if nlp_results.get("keywords"):
                kw_rows = ""
                for kw in nlp_results["keywords"][:15]:
                    kw_rows += f"<tr><td>{kw['keyword']}</td><td>{kw['count']:,}</td><td>{kw['frequency']}%</td></tr>"
                nlp_html += f"""
                <div class="section">
                    <h2>🔑 Top Keywords</h2>
                    <table>
                        <thead><tr><th>Keyword</th><th>Count</th><th>Frequency</th></tr></thead>
                        <tbody>{kw_rows}</tbody>
                    </table>
                </div>"""

        # Top endpoints
        top_endpoints_html = ""
        if "top_endpoints" in aggregations:
            rows = ""
            for ep, cnt in list(aggregations["top_endpoints"].items())[:10]:
                rows += f"<tr><td>{ep}</td><td>{cnt:,}</td></tr>"
            top_endpoints_html = f"""
            <div class="section">
                <h2>🌐 Top Endpoints</h2>
                <table>
                    <thead><tr><th>Endpoint</th><th>Requests</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Analysis Report - {report_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 2rem;
            margin-bottom: 2rem;
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{ font-size: 2rem; color: #00d2ff; margin-bottom: 0.5rem; }}
        .header p {{ color: #8892b0; }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s;
        }}
        .metric-card:hover {{ transform: translateY(-3px); }}
        .metric-card .value {{ font-size: 2rem; font-weight: 700; color: #00d2ff; }}
        .metric-card .label {{ color: #8892b0; margin-top: 0.3rem; font-size: 0.9rem; }}
        .metric-card.danger .value {{ color: #f44336; }}
        .metric-card.warning .value {{ color: #ff9800; }}
        .section {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .section h2 {{ color: #00d2ff; margin-bottom: 1rem; font-size: 1.3rem; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        th {{ color: #00d2ff; font-weight: 600; }}
        .msg-cell {{ max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.85rem; color: #8892b0; }}
        .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
        .chart-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .footer {{ text-align: center; color: #555; padding: 2rem; font-size: 0.85rem; }}
        ul {{ padding-left: 1.5rem; }} li {{ margin-bottom: 0.4rem; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Log Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {report_name}</p>
        </div>

        <div class="metrics">
            <div class="metric-card">
                <div class="value">{total:,}</div>
                <div class="label">Total Records</div>
            </div>
            <div class="metric-card {'danger' if error_rate > 10 else 'warning' if error_rate > 5 else ''}">
                <div class="value">{error_rate}%</div>
                <div class="label">Error Rate ({error_count:,} errors)</div>
            </div>
            <div class="metric-card {'danger' if anomaly_rate > 10 else 'warning' if anomaly_rate > 5 else ''}">
                <div class="value">{anomaly_count:,}</div>
                <div class="label">Anomalies ({anomaly_rate}%)</div>
            </div>
            <div class="metric-card">
                <div class="value">{pattern_count}</div>
                <div class="label">Patterns Found</div>
            </div>
        </div>

        <div class="charts">
            {charts_html}
        </div>

        {top_endpoints_html}
        {pattern_html}
        {nlp_html}

        <div class="footer">
            <p>AI-DDE Log Analyzer v1.0 - Batch Processing Report</p>
        </div>
    </div>
</body>
</html>"""
        return html
