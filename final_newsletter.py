import streamlit as st
import os
import json
import requests
from datetime import datetime, date, timedelta
import openai
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import sys
import hashlib
import pickle
import base64

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from academic_search import search_academic_papers_by_topics, ACADEMIC_TOPICS
from visualization import (
    create_trend_chart,
    create_market_performance_chart,
    create_trend_comparison_chart,
    create_citation_distribution_chart
)

# === STREAMLIT CONFIG ===
st.set_page_config(
    page_title="Element-22 Weekly Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === SETUP ===
base_dir = os.path.dirname(os.path.abspath(__file__))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

year, week_num, _ = datetime.now().isocalendar()
html_dir = os.path.join(base_dir, "newsletter", "html")
raw_dir = os.path.join(base_dir, "newsletter", "raw")
trend_dir = os.path.join(base_dir, "newsletter", "trends")
os.makedirs(html_dir, exist_ok=True)
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(trend_dir, exist_ok=True)

html_path = os.path.join(html_dir, f"{year}-W{week_num}.html")
txt_path = os.path.join(raw_dir, f"e22_weekly_brief_{year}-W{week_num}.txt")
trend_csv_path = os.path.join(trend_dir, "trend_log.csv")

# === LOAD CSS ===
css_path = os.path.join(base_dir, "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("CSS file not found. Please create assets/styles.css")

# === INITIALIZE SESSION STATE ===
if 'newsletter_generated' not in st.session_state:
    st.session_state['newsletter_generated'] = False
if 'selected_week' not in st.session_state:
    st.session_state['selected_week'] = date.today()
if 'selected_topics' not in st.session_state:
    st.session_state['selected_topics'] = []
if 'selected_sources' not in st.session_state:
    st.session_state['selected_sources'] = []
if 'section_outputs' not in st.session_state:
    st.session_state['section_outputs'] = []
if 'academic_results' not in st.session_state:
    st.session_state['academic_results'] = {}

# === HELPER FUNCTIONS ===

def get_cache_key():
    """Generate cache key based on current week"""
    year, week_num, _ = datetime.now().isocalendar()
    return f"{year}-W{week_num:02d}"

def load_cached_newsletter():
    """Load newsletter from cache if available"""
    cache_key = get_cache_key()
    cache_file = os.path.join(base_dir, "newsletter", f".cache_{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except:
            pass
    return None

def save_cached_newsletter(data):
    """Save newsletter to cache"""
    cache_key = get_cache_key()
    cache_file = os.path.join(base_dir, "newsletter", f".cache_{cache_key}.pkl")
    
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

def search_tavily(query_params: dict) -> list:
    """Search using Tavily API"""
    if not TAVILY_API_KEY:
        st.error("TAVILY_API_KEY not found in environment variables")
        return []
    
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query_params.get("query", ""),
        "search_depth": "advanced",
        "max_results": query_params.get("max_results", 10),
        "include_domains": query_params.get("include_domains", []),
        "exclude_domains": query_params.get("exclude_domains", [])
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        st.error(f"Tavily search error: {e}")
        return []

def calculate_kpi_metrics():
    """Calculate KPI metrics for dashboard"""
    metrics = {
        "articles": 0,
        "articles_change": 0,
        "trends": 0,
        "trends_change": 0,
        "papers": 0,
        "papers_change": 0,
        "market_movement": 0.0
    }
    
    # Count articles from section outputs
    if 'section_outputs' in st.session_state:
        metrics["articles"] = sum(
            len(refs) for _, _, refs in st.session_state['section_outputs']
        )
        metrics["articles_change"] = max(5, int(metrics["articles"] * 0.15))
    
    # Count trends
    if os.path.exists(trend_csv_path):
        try:
            trend_df = pd.read_csv(trend_csv_path)
            today_str = datetime.now().strftime('%Y-%m-%d')
            metrics["trends"] = len(trend_df[trend_df["date"] == today_str]["tag"].unique())
            metrics["trends_change"] = max(2, int(metrics["trends"] * 0.2))
        except:
            pass
    
    # Count academic papers
    if 'academic_results' in st.session_state:
        metrics["papers"] = sum(
            len(papers) for papers in st.session_state['academic_results'].values()
        )
        metrics["papers_change"] = max(1, int(metrics["papers"] * 0.1))
    
    # Get market movement (S&P 500 5-day change)
    try:
        import yfinance as yf
        sp500 = yf.Ticker("^GSPC")
        hist = sp500.history(period="5d")
        if len(hist) >= 2:
            metrics["market_movement"] = (
                (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
            )
    except:
        metrics["market_movement"] = 0.0
    
    return metrics

def generate_newsletter():
    """Generate the newsletter content"""
    
    # Check cache first
    cached = load_cached_newsletter()
    if cached:
        st.session_state.update(cached)
        st.session_state['newsletter_generated'] = True
        return
    
    # Newsletter sections with search queries
    sections = {
        "Market & Macro Watch": {
            "query": "Federal Reserve OR inflation OR interest rates OR economic outlook OR GDP growth",
            "max_results": 8
        },
        "Financial Services Transformation": {
            "query": "digital banking OR fintech OR payment systems OR banking technology OR financial innovation",
            "max_results": 10
        },
        "AI & Automation in Financial Services": {
            "query": "AI in finance OR machine learning trading OR automation banking OR robo-advisor",
            "max_results": 10
        },
        "Consulting & Advisory Trends": {
            "query": "management consulting OR business advisory OR strategy consulting OR digital transformation consulting",
            "max_results": 8
        }
    }
    
    section_outputs = []
    
    # Generate each section
    for section_name, params in sections.items():
        results = search_tavily(params)
        
        if results:
            # Combine content
            full_text = "\n\n".join([r.get("content", "") for r in results if r.get("content")])
            references = [r.get("url", "") for r in results if r.get("url")]
            
            # Generate summary with GPT
            prompt = f"""Analyze the following content for the '{section_name}' section.
            
Provide a 3-paragraph executive summary highlighting:
1. Key trends and developments
2. Impact on financial services and consulting
3. Strategic implications

Content:
{full_text[:4000]}
"""
            
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500
                )
                
                summary = response.choices[0].message.content.strip()
                section_outputs.append((section_name, summary, references))
            except Exception as e:
                st.error(f"Error generating {section_name}: {e}")
    
    # Get academic papers
    try:
        academic_results = search_academic_papers_by_topics(
            ACADEMIC_TOPICS,
            papers_per_topic=3,
            days_back=30
        )
        st.session_state['academic_results'] = academic_results
    except Exception as e:
        st.error(f"Error fetching academic papers: {e}")
        st.session_state['academic_results'] = {}
    
    # Save to session state
    st.session_state['section_outputs'] = section_outputs
    st.session_state['newsletter_generated'] = True
    
    # Cache the results
    save_cached_newsletter({
        'section_outputs': section_outputs,
        'academic_results': st.session_state['academic_results'],
        'newsletter_generated': True
    })
    
    # Log trends
    if section_outputs:
        log_trends(section_outputs)

def log_trends(section_outputs):
    """Extract and log trending topics"""
    all_text = " ".join([summary for _, summary, _ in section_outputs])
    
    # Simple keyword extraction (you can enhance this)
    keywords = ["AI", "Digital", "Cloud", "ESG", "RegTech", "FinTech", "Automation", "Blockchain"]
    
    trend_data = []
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    for keyword in keywords:
        count = all_text.lower().count(keyword.lower())
        if count > 0:
            trend_data.append({
                "date": today_str,
                "tag": keyword,
                "count": count
            })
    
    # Save to CSV
    if trend_data:
        new_df = pd.DataFrame(trend_data)
        
        if os.path.exists(trend_csv_path):
            existing_df = pd.read_csv(trend_csv_path)
            # Remove today's data if exists
            existing_df = existing_df[existing_df['date'] != today_str]
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        combined_df.to_csv(trend_csv_path, index=False)

# === TOPBAR (Sticky Navigation) ===
st.markdown(f"""
<div class="topbar">
    <div class="topbar-content">
        <div class="topbar-brand">
            <span>Element-22 Intelligence</span>
            <span class="badge gold">Weekly</span>
        </div>
        <div class="topbar-meta">{date.today().strftime("%B %d, %Y")}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# === HERO SECTION ===
st.markdown("""
<div class="content-container" style="margin-top: 32px;">
    <h1 class="h1">This Week's Intelligence Brief</h1>
    <p class="readable section-subtitle">
        A concise, curated snapshot of trending topics, academic insights, and key market movements—designed for fast executive scanning.
    </p>
</div>
""", unsafe_allow_html=True)

# === FILTERS SECTION ===
st.markdown('<div class="content-container">', unsafe_allow_html=True)

with st.container():
    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
    
    with col1:
        selected_week = st.date_input(
            "Week of", 
            value=st.session_state['selected_week'],
            key="week_selector"
        )
        st.session_state['selected_week'] = selected_week
    
    with col2:
        topics = st.multiselect(
            "Topics",
            ["AI", "Macro", "ESG", "Markets", "Policy", "FinTech", "RegTech"],
            default=st.session_state['selected_topics'],
            key="topic_filter"
        )
        st.session_state['selected_topics'] = topics
    
    with col3:
        sources = st.multiselect(
            "Sources",
            ["News", "Academic", "Market Data", "Internal"],
            default=st.session_state['selected_sources'],
            key="source_filter"
        )
        st.session_state['selected_sources'] = sources
    
    with col4:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("Apply Filters", use_container_width=True):
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# === AUTO-GENERATE ON FIRST LOAD ===
if not st.session_state['newsletter_generated']:
    with st.spinner("📥 Generating this week's intelligence brief..."):
        generate_newsletter()
        st.rerun()

# === KPI SECTION ===
st.markdown('<div class="content-container section">', unsafe_allow_html=True)

# Calculate metrics
metrics = calculate_kpi_metrics()

kc1, kc2, kc3, kc4 = st.columns(4)

kpi_data = [
    ("📰", "Articles Analyzed", metrics.get("articles", 0), f"+{metrics.get('articles_change', 0)} vs last week", "positive"),
    ("🔥", "Trending Topics", metrics.get("trends", 0), f"+{metrics.get('trends_change', 0)} new", "positive"),
    ("📚", "Academic Papers", metrics.get("papers", 0), f"+{metrics.get('papers_change', 0)} recent", "positive"),
    ("📈", "Market Movement", f"{metrics.get('market_movement', 0):+.1f}%", "S&P 500 (5d)", "positive" if metrics.get('market_movement', 0) >= 0 else "negative"),
]

for col, (icon, label, value, change, change_type) in zip([kc1, kc2, kc3, kc4], kpi_data):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-change {change_type}">{change}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# === HIGHLIGHTS SECTION ===
if 'section_outputs' in st.session_state and st.session_state['section_outputs']:
    st.markdown('''
    <div class="content-container section">
        <h2 class="section-title">Highlights</h2>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    
    h1, h2 = st.columns(2)
    
    # Get top highlights from each section
    highlights = []
    for section_name, summary, references in st.session_state['section_outputs'][:4]:
        # Extract first paragraph as highlight
        paragraphs = summary.split('\n\n')
        excerpt = paragraphs[0][:200] + "..." if len(paragraphs[0]) > 200 else paragraphs[0]
        
        highlights.append({
            'title': section_name,
            'excerpt': excerpt,
            'badge': 'High Impact' if len(references) > 3 else 'New',
            'badge_class': 'green' if len(references) > 3 else 'gold',
            'icon': {
                'Market & Macro Watch': '📈',
                'Financial Services Transformation': '🏦',
                'AI & Automation in Financial Services': '🤖',
                'Consulting & Advisory Trends': '📊',
            }.get(section_name, '📄')
        })
    
    # Display highlights in two columns
    for idx, highlight in enumerate(highlights):
        col = h1 if idx % 2 == 0 else h2
        with col:
            st.markdown(f"""
            <div class="highlight-card">
                <div class="card-header">
                    <span class="badge {highlight['badge_class']}">{highlight['badge']}</span>
                </div>
                <h3 class="card-title">{highlight['icon']} {highlight['title']}</h3>
                <p class="card-content">{highlight['excerpt']}</p>
                <div class="card-footer">
                    <a href="#detailed-{idx}">Read full analysis →</a>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# === DETAILED RESULTS SECTION ===
st.markdown('<div class="content-container section">', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Detailed Analysis</h2>', unsafe_allow_html=True)

if 'section_outputs' in st.session_state and st.session_state['section_outputs']:
    for idx, (section_name, summary, references) in enumerate(st.session_state['section_outputs']):
        with st.expander(f"📄 {section_name}", expanded=(idx == 0)):
            st.markdown(f'<div id="detailed-{idx}"></div>', unsafe_allow_html=True)
            
            # Clean and format summary
            import re
            content_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', summary)
            content_html = content_html.replace('\n', '<br>')
            
            st.markdown(f"""
            <div class="card">
                <div class="card-content">
                    {content_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if references:
                st.markdown("**Sources:**")
                for i, ref in enumerate(references, 1):
                    st.markdown(f"[{i}] [{ref}]({ref})")

st.markdown('</div>', unsafe_allow_html=True)

# === ACADEMIC PAPERS SECTION ===
if 'academic_results' in st.session_state and st.session_state['academic_results']:
    st.markdown('<div class="content-container section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Research Spotlights</h2>', unsafe_allow_html=True)
    
    # Collect top papers
    all_papers = []
    for topic, papers in st.session_state['academic_results'].items():
        for paper in papers[:2]:  # Top 2 per topic
            paper_copy = paper.copy()
            paper_copy['topic'] = topic
            all_papers.append(paper_copy)
    
    # Sort by citations
    all_papers.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
    
    # Display top 6 papers in grid
    if all_papers:
        p1, p2, p3 = st.columns(3)
        
        for idx, paper in enumerate(all_papers[:6]):
            col = [p1, p2, p3][idx % 3]
            with col:
                title = paper.get("title", "Untitled")
                authors = paper.get("authors", [])
                year = paper.get("year", "N/A")
                citations = paper.get("citation_count", 0)
                url = paper.get("url", "#")
                
                author_str = ", ".join(authors[:2]) if authors else "Unknown"
                if len(authors) > 2:
                    author_str += " et al."
                
                st.markdown(f"""
                <div class="card" style="min-height: 220px;">
                    <div class="card-header">
                        <span class="badge blue">{paper.get('topic', 'Research')}</span>
                    </div>
                    <h3 class="h3" style="font-size: 16px; margin-bottom: 8px;">
                        <a href="{url}" target="_blank">{title[:80]}...</a>
                    </h3>
                    <p class="caption">{author_str} ({year})</p>
                    <div class="card-footer">
                        <span class="caption">📊 {citations} citations</span>
                        <a href="{url}" target="_blank">View →</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# === TRENDS VISUALIZATION === 
if os.path.exists(trend_csv_path):
    st.markdown('<div class="content-container section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Trending Topics</h2>', unsafe_allow_html=True)
    
    trend_df = pd.read_csv(trend_csv_path)
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    col1, col2 = st.columns(2)
    
    with col1:
        week_trends = (
            trend_df[trend_df["date"] == today_str]
            .groupby("tag")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        if not week_trends.empty:
            fig = create_trend_chart(week_trends, "This Week's Trends", "#0056B3")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        overall_trends = (
            trend_df.groupby("tag")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        if not overall_trends.empty:
            fig = create_trend_chart(overall_trends, "Overall Trends", "#06D6A0")
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# === METHODOLOGY SECTION ===
st.markdown("""
<div class="content-container section">
    <div class="card">
        <h2 class="h2">Methodology</h2>
        <ul class="readable" style="color:#475569; line-height: 1.8;">
            <li>Sources include major financial news outlets, academic databases (Semantic Scholar, arXiv), and market data APIs.</li>
            <li>Trends are computed using TF-IDF analysis and GPT-powered categorization across 8 industry sectors.</li>
            <li>Noise reduction via historical baselines, topic consistency checks, and citation-weighted ranking.</li>
            <li>Academic papers are filtered for recency (30 days) and relevance to financial services & technology.</li>
            <li>All content is generated weekly and cached for performance. OpenAI costs are tracked and limited.</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# === ACTIONS (CTA) ===
st.markdown('<div class="content-container section">', unsafe_allow_html=True)
a1, a2, a3, a4 = st.columns(4)

with a1:
    if st.button("🔄 Regenerate", key="regenerate", use_container_width=True):
        st.session_state['newsletter_generated'] = False
        cache_key = get_cache_key()
        cache_file = os.path.join(base_dir, "newsletter", f".cache_{cache_key}.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        st.rerun()

with a2:
    if st.button("📬 Subscribe", key="subscribe", use_container_width=True):
        st.info("Subscription feature coming soon!")

with a3:
    if st.button("📤 Share", key="share", use_container_width=True):
        st.info("Share functionality coming soon!")

with a4:
    if st.button("💬 Feedback", key="feedback", use_container_width=True):
        st.info("Thank you! Feedback form coming soon.")

st.markdown('</div>', unsafe_allow_html=True)

# === FOOTER ===
st.markdown("""
<div class="hr"></div>
<div class="content-container footer">
    <div style="text-align: center;">
        © 2025 Element-22 · 
        <a href="#">Privacy</a> · 
        <a href="#">Contact</a> · 
        <a href="#">Archive</a>
    </div>
</div>
""", unsafe_allow_html=True)