"""
Reusable Plotly chart components for the dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional


DARK_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0e0e0", family="Inter"),
)

COLORS = {
    "primary": "#00d2ff",
    "success": "#4caf50",
    "warning": "#ff9800",
    "danger": "#f44336",
    "purple": "#9c27b0",
    "blue": "#3a7bd5",
    "gradient": ["#00d2ff", "#4caf50", "#ff9800", "#f44336", "#9c27b0"],
}


def create_pie_chart(labels: List[str], values: List[int], title: str, height: int = 350) -> go.Figure:
    """Create a styled donut chart."""
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=COLORS["gradient"][:len(labels)]),
    )])
    fig.update_layout(title=title, height=height, **DARK_THEME)
    return fig


def create_bar_chart(x: list, y: list, title: str, color: str = None, height: int = 350) -> go.Figure:
    """Create a styled bar chart."""
    fig = go.Figure(data=[go.Bar(
        x=x, y=y,
        marker_color=color or COLORS["primary"],
        marker_line_width=0,
    )])
    fig.update_layout(title=title, height=height, **DARK_THEME)
    return fig


def create_line_chart(x: list, y: list, title: str, color: str = None, fill: bool = True, height: int = 350) -> go.Figure:
    """Create a styled line chart with optional fill."""
    line_color = color or COLORS["success"]
    fig = go.Figure(data=[go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        line=dict(color=line_color, width=2),
        marker=dict(size=5),
        fill="tozeroy" if fill else None,
        fillcolor=f"rgba({int(line_color[1:3],16)},{int(line_color[3:5],16)},{int(line_color[5:7],16)},0.1)" if fill else None,
    )])
    fig.update_layout(title=title, height=height, **DARK_THEME)
    return fig


def create_heatmap(z: list, x: list, y: list, title: str, height: int = 400) -> go.Figure:
    """Create a styled heatmap."""
    fig = go.Figure(data=[go.Heatmap(
        z=z, x=x, y=y,
        colorscale="Viridis",
    )])
    fig.update_layout(title=title, height=height, **DARK_THEME)
    return fig


def create_histogram(values: list, title: str, color: str = None, nbins: int = 50, height: int = 350) -> go.Figure:
    """Create a styled histogram."""
    fig = go.Figure(data=[go.Histogram(
        x=values,
        nbinsx=nbins,
        marker_color=color or COLORS["primary"],
    )])
    fig.update_layout(title=title, height=height, **DARK_THEME)
    return fig
