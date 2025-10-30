"""
Visualization Module
Creates interactive Plotly charts for the newsletter
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import yfinance as yf
from typing import Dict, List, Optional

def create_trend_chart(
    trend_data: pd.Series,
    title: str = "Trend Analysis",
    color: str = "#0056b3"
) -> go.Figure:
    """
    Create an interactive bar chart for trends
    
    Args:
        trend_data: Pandas Series with trend tags as index and counts as values
        title: Chart title
        color: Bar color
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=[
        go.Bar(
            x=trend_data.index,
            y=trend_data.values,
            marker_color=color,
            hovertemplate='<b>%{x}</b><br>Mentions: %{y}<extra></extra>',
            text=trend_data.values,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#0056b3', 'family': 'Arial'}
        },
        xaxis_title="Trend Tag",
        yaxis_title="Mentions",
        template="plotly_white",
        height=400,
        hovermode='x',
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=100),
        xaxis={
            'tickangle': -45,
            'tickfont': {'size': 11}
        }
    )
    
    return fig


def create_market_performance_chart(
    start_date: Optional[str] = None
) -> go.Figure:
    """
    Create YTD performance chart for major market indices
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format (defaults to start of current year)
        
    Returns:
        Plotly Figure object with normalized performance
    """
    if start_date is None:
        start_date = f"{datetime.now().year}-01-01"
    
    # Define tickers and colors
    tickers = {
        "^GSPC": {"name": "S&P 500", "color": "#3366cc"},
        "^IXIC": {"name": "NASDAQ", "color": "#dc3912"},
        "^DJI": {"name": "Dow Jones", "color": "#109618"}
    }
    
    fig = go.Figure()
    has_data = False
    
    for ticker, info in tickers.items():
        try:
            # Download data with proper parameters
            data = yf.download(
                ticker,
                start=start_date,
                end=datetime.now().strftime('%Y-%m-%d'),
                interval="1d",
                progress=False,
                show_errors=False
            )
            
            if not data.empty and len(data) > 0:
                # Handle multi-index columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    close_col = ('Close', ticker)
                    if close_col in data.columns:
                        close_data = data[close_col]
                    else:
                        close_data = data['Close']
                else:
                    close_data = data['Close']
                
                # Remove any NaN values
                close_data = close_data.dropna()
                
                if len(close_data) > 0:
                    # Normalize to 100 at start
                    normalized = (close_data / close_data.iloc[0]) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=normalized.index,
                        y=normalized.values,
                        mode='lines',
                        name=info["name"],
                        line=dict(color=info["color"], width=2.5),
                        hovertemplate=(
                            f'<b>{info["name"]}</b><br>' +
                            'Date: %{x|%Y-%m-%d}<br>' +
                            'Performance: %{y:.2f}%<extra></extra>'
                        )
                    ))
                    has_data = True
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue
    
    if not has_data:
        # Create a placeholder figure
        fig.add_annotation(
            text="Market data temporarily unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='#6c757d')
        )
    
    fig.update_layout(
        title={
            'text': 'YTD Market Performance (Normalized)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#0056b3', 'family': 'Arial'}
        },
        xaxis_title="Date",
        yaxis_title="Normalized Price (Start = 100)",
        template="plotly_white",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    return fig


def create_trend_comparison_chart(
    df: pd.DataFrame,
    current_week: str
) -> go.Figure:
    """
    Create side-by-side comparison of this week vs overall trends
    
    Args:
        df: DataFrame with columns ['date', 'tag', 'count']
        current_week: Date string for current week (YYYY-MM-DD)
        
    Returns:
        Plotly Figure with subplots
    """
    from plotly.subplots import make_subplots
    
    # Get top trends for current week
    week_trends = (
        df[df['date'] == current_week]
        .groupby('tag')['count']
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    
    # Get top overall trends
    overall_trends = (
        df.groupby('tag')['count']
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top Trends This Week", "Top Trends Overall"),
        horizontal_spacing=0.15
    )
    
    # Add week trends
    fig.add_trace(
        go.Bar(
            x=week_trends.index,
            y=week_trends.values,
            marker_color='#1f77b4',
            name='This Week',
            hovertemplate='<b>%{x}</b><br>Mentions: %{y}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add overall trends
    fig.add_trace(
        go.Bar(
            x=overall_trends.index,
            y=overall_trends.values,
            marker_color='#2ca02c',
            name='Overall',
            hovertemplate='<b>%{x}</b><br>Mentions: %{y}<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)
    
    fig.update_layout(
        height=450,
        template="plotly_white",
        showlegend=False,
        title_text="Trend Analysis Dashboard",
        title_x=0.5,
        title_font=dict(size=20, color='#0056b3', family='Arial')
    )
    
    return fig


def create_trend_timeline(
    df: pd.DataFrame,
    top_n: int = 5
) -> go.Figure:
    """
    Create timeline showing trend evolution over weeks
    
    Args:
        df: DataFrame with columns ['date', 'tag', 'count']
        top_n: Number of top trends to show
        
    Returns:
        Plotly Figure with line chart
    """
    # Get top tags overall
    top_tags = (
        df.groupby('tag')['count']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    
    # Filter for top tags
    filtered_df = df[df['tag'].isin(top_tags)]
    
    # Pivot for timeline
    timeline = filtered_df.pivot_table(
        index='date',
        columns='tag',
        values='count',
        aggfunc='sum',
        fill_value=0
    )
    
    fig = go.Figure()
    
    colors = ['#0056b3', '#3a86ff', '#ffb703', '#06d6a0', '#fb8500']
    
    for i, tag in enumerate(timeline.columns):
        fig.add_trace(go.Scatter(
            x=timeline.index,
            y=timeline[tag],
            mode='lines+markers',
            name=tag,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8),
            hovertemplate=(
                f'<b>{tag}</b><br>' +
                'Date: %{x}<br>' +
                'Mentions: %{y}<extra></extra>'
            )
        ))
    
    fig.update_layout(
        title={
            'text': 'Trend Evolution Over Time',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#0056b3', 'family': 'Arial'}
        },
        xaxis_title="Week",
        yaxis_title="Mentions",
        template="plotly_white",
        height=450,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=50, r=150, t=80, b=50)
    )
    
    return fig


def create_citation_distribution_chart(
    papers: List[Dict]
) -> go.Figure:
    """
    Create histogram of citation counts for academic papers
    
    Args:
        papers: List of paper dictionaries with 'citation_count' field
        
    Returns:
        Plotly Figure with histogram
    """
    citation_counts = [p.get('citation_count', 0) for p in papers if p.get('citation_count', 0) > 0]
    
    if not citation_counts:
        # Return empty figure if no citations
        fig = go.Figure()
        fig.add_annotation(
            text="No citation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='#6c757d')
        )
        return fig
    
    fig = go.Figure(data=[
        go.Histogram(
            x=citation_counts,
            nbinsx=20,
            marker_color='#ffb703',
            hovertemplate='Citations: %{x}<br>Papers: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Citation Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#0056b3', 'family': 'Arial'}
        },
        xaxis_title="Citation Count",
        yaxis_title="Number of Papers",
        template="plotly_white",
        height=400,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def create_kpi_chart(
    values: Dict[str, int],
    title: str = "Key Metrics"
) -> go.Figure:
    """
    Create a simple indicator chart for KPIs
    
    Args:
        values: Dictionary of {metric_name: value}
        title: Chart title
        
    Returns:
        Plotly Figure with indicators
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(values.keys()),
        y=list(values.values()),
        marker_color=['#0056b3', '#3a86ff', '#ffb703', '#06d6a0'],
        text=list(values.values()),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#0056b3', 'family': 'Arial'}
        },
        template="plotly_white",
        height=350,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=100),
        xaxis={'tickangle': -45}
    )
    
    return fig


if __name__ == "__main__":
    # Test the visualization module
    print("Testing Visualization Module...\n")
    
    # Test trend chart
    print("Creating sample trend chart...")
    sample_trends = pd.Series({
        "AI in Finance": 15,
        "Cloud Migration": 12,
        "RegTech": 10,
        "Data Governance": 8,
        "ESG": 6
    })
    fig = create_trend_chart(sample_trends, "Sample Trends")
    print("✓ Trend chart created\n")
    
    # Test market performance chart
    print("Creating market performance chart...")
    try:
        fig = create_market_performance_chart()
        print("✓ Market chart created\n")
    except Exception as e:
        print(f"✗ Market chart failed: {e}\n")
    
    print("Visualization module test complete!")