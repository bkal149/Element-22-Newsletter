import streamlit as st
import os
import json
import requests
from datetime import datetime
import openai
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import sys

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from academic_search import search_academic_papers_by_topics, ACADEMIC_TOPICS
from visualization import (
    create_trend_chart,
    create_market_performance_chart,
    create_trend_comparison_chart,
    create_citation_distribution_chart
)

# === COST CONTROL CONFIGURATION ===
COST_CONTROLS = {
    "max_tokens_per_request": 2000,  # Max tokens in completion
    "max_total_tokens_per_run": 50000,  # Max total tokens per newsletter generation
    "max_api_calls_per_run": 20,  # Max OpenAI API calls
    "use_gpt35_for_trends": True,  # Use cheaper GPT-3.5 for trend extraction
    "use_gpt4_for_summaries": False,  # Set to False to use GPT-3.5 for summaries (cheaper)
    "cache_enabled": True,  # Enable caching to avoid duplicate API calls
    "temperature": 0.3,  # Lower temperature = more deterministic = less tokens
}

# Token tracking
if 'openai_usage' not in st.session_state:
    st.session_state['openai_usage'] = {
        'total_tokens': 0,
        'api_calls': 0,
        'estimated_cost': 0.0,
        'calls_log': []
    }

# Pricing (as of late 2024, in USD per 1K tokens)
GPT4_INPUT_PRICE = 0.03
GPT4_OUTPUT_PRICE = 0.06
GPT35_INPUT_PRICE = 0.0005
GPT35_OUTPUT_PRICE = 0.0015

def track_openai_usage(model: str, prompt_tokens: int, completion_tokens: int, purpose: str):
    """Track OpenAI API usage and costs"""
    total = prompt_tokens + completion_tokens
    
    # Calculate cost
    if "gpt-4" in model:
        cost = (prompt_tokens * GPT4_INPUT_PRICE / 1000) + (completion_tokens * GPT4_OUTPUT_PRICE / 1000)
    else:
        cost = (prompt_tokens * GPT35_INPUT_PRICE / 1000) + (completion_tokens * GPT35_OUTPUT_PRICE / 1000)
    
    # Update session state
    st.session_state['openai_usage']['total_tokens'] += total
    st.session_state['openai_usage']['api_calls'] += 1
    st.session_state['openai_usage']['estimated_cost'] += cost
    st.session_state['openai_usage']['calls_log'].append({
        'time': datetime.now().isoformat(),
        'model': model,
        'purpose': purpose,
        'tokens': total,
        'cost': cost
    })
    
    return total, cost

def check_cost_limits():
    """Check if we've exceeded cost limits"""
    usage = st.session_state['openai_usage']
    
    if usage['total_tokens'] >= COST_CONTROLS['max_total_tokens_per_run']:
        st.warning(f"⚠️ Token limit reached ({usage['total_tokens']:,} tokens). Some content may be truncated.")
        return False
    
    if usage['api_calls'] >= COST_CONTROLS['max_api_calls_per_run']:
        st.warning(f"⚠️ API call limit reached ({usage['api_calls']} calls). Some sections may be skipped.")
        return False
    
    return True

def display_cost_dashboard():
    """Display cost tracking dashboard in sidebar"""
    usage = st.session_state['openai_usage']
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 OpenAI Usage")
    st.sidebar.metric("Total Tokens", f"{usage['total_tokens']:,}")
    st.sidebar.metric("API Calls", usage['api_calls'])
    st.sidebar.metric("Est. Cost", f"${usage['estimated_cost']:.4f}")
    
    if usage['api_calls'] > 0:
        with st.sidebar.expander("📊 Usage Details"):
            for log in usage['calls_log'][-5:]:  # Show last 5 calls
                st.text(f"{log['purpose'][:20]}: {log['tokens']} tokens (${log['cost']:.4f})")

# === FIRST STREAMLIT COMMAND ===
st.set_page_config(
    page_title="E22 Weekly Brief",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# === LOAD CUSTOM CSS ===
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# === SETUP ===
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

year, week_num, _ = datetime.now().isocalendar()
base_dir = os.path.dirname(os.path.abspath(__file__))
html_dir = os.path.join(base_dir, "newsletter", "html")
raw_dir = os.path.join(base_dir, "newsletter", "raw")
trend_dir = os.path.join(base_dir, "newsletter", "trends")
os.makedirs(html_dir, exist_ok=True)
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(trend_dir, exist_ok=True)

html_path = os.path.join(html_dir, f"{year}-W{week_num}.html")
txt_path = os.path.join(raw_dir, f"e22_weekly_brief_{year}-W{week_num}.txt")
trend_csv_path = os.path.join(trend_dir, "trend_log.csv")

# === HELPER FUNCTIONS ===

def render_hero_header():
    """Render the gradient hero header"""
    today = datetime.now().strftime('%B %d, %Y')
    st.markdown(f"""
    <div class="hero-header">
        <h1>🗞️ E22 Weekly Brief</h1>
        <p>Your Strategic Intelligence Digest</p>
        <p class="subtitle">Week {week_num}, {year} • {today}</p>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_dashboard(metrics: dict):
    """Render KPI dashboard with cards"""
    st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
    
    kpis = [
        {
            "icon": "📰",
            "value": metrics.get("articles", 0),
            "label": "Articles Analyzed",
            "change": metrics.get("articles_change", 0),
            "card_type": "info"
        },
        {
            "icon": "🔥",
            "value": metrics.get("trends", 0),
            "label": "Trending Topics",
            "change": metrics.get("trends_change", 0),
            "card_type": "accent"
        },
        {
            "icon": "📚",
            "value": metrics.get("papers", 0),
            "label": "Academic Papers",
            "change": metrics.get("papers_change", 0),
            "card_type": "success"
        },
        {
            "icon": "📈",
            "value": f"{metrics.get('market_movement', 0):+.1f}%",
            "label": "S&P 500 (Week)",
            "change": None,
            "card_type": "warning"
        }
    ]
    
    cols = st.columns(4)
    for col, kpi in zip(cols, kpis):
        with col:
            change_html = ""
            if kpi["change"] is not None:
                change_class = "positive" if kpi["change"] >= 0 else "negative"
                change_symbol = "↑" if kpi["change"] >= 0 else "↓"
                change_html = f'<div class="kpi-change {change_class}">{change_symbol} {abs(kpi["change"])} vs last week</div>'
            
            st.markdown(f"""
            <div class="kpi-card {kpi['card_type']}">
                <div class="kpi-header">
                    <span class="kpi-icon">{kpi['icon']}</span>
                </div>
                <div class="kpi-value">{kpi['value']}</div>
                <div class="kpi-label">{kpi['label']}</div>
                {change_html}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_section_card(title: str, content: str, icon: str, references: list = None):
    """Render a content section as a card"""
    st.markdown(f"""
    <div class="section-card">
        <div class="section-header">
            <span class="section-icon">{icon}</span>
            <h2 class="section-title">{title}</h2>
        </div>
        <div class="section-content">
            {content}
        </div>
    """, unsafe_allow_html=True)
    
    if references:
        with st.expander(f"📎 View {len(references)} Sources"):
            for i, ref in enumerate(references, 1):
                st.markdown(f"[{i}] [{ref}]({ref})")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_academic_paper_card(paper: dict):
    """Render a single academic paper card"""
    authors_str = ", ".join(paper["authors"][:3])
    if len(paper["authors"]) > 3:
        authors_str += f" et al. ({len(paper['authors'])} authors)"
    
    abstract_preview = paper["abstract"][:300] + "..." if len(paper["abstract"]) > 300 else paper["abstract"]
    
    # Build paper links
    links_html = ""
    if paper.get("url"):
        links_html += f'<a href="{paper["url"]}" target="_blank" class="paper-link-btn">🔗 View Paper</a>'
    if paper.get("arxiv_id"):
        links_html += f'<a href="https://arxiv.org/abs/{paper["arxiv_id"]}" target="_blank" class="paper-link-btn">📄 arXiv</a>'
    
    st.markdown(f"""
    <div class="paper-card">
        <div class="paper-title">
            <a href="{paper['url']}" target="_blank">{paper['title']}</a>
        </div>
        <div class="paper-meta">
            <span>👥 {authors_str}</span>
            <span>📅 {paper['year']}</span>
            <span class="badge citation">📊 {paper['citation_count']} citations</span>
            <span class="badge venue">🏛️ {paper['venue']}</span>
        </div>
        <div class="paper-abstract">
            {abstract_preview}
        </div>
        <div class="paper-links">
            {links_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# === CORE FUNCTIONS WITH COST CONTROLS ===

def search_tavily(params):
    """Search using Tavily API"""
    try:
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
        payload = {
            "query": params["query"],
            "search_depth": "advanced",
            "topic": params.get("topic", "news"),
            "time_range": params.get("time_range", "week"),
            "max_results": params.get("max_results", 10),
            "include_answer": "advanced",
            "include_content": True,
            "include_titles": True
        }
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        st.error(f"❌ Tavily query failed: {e}")
        return []


def extract_trends(text):
    """Extract trend tags using GPT (with cost controls)"""
    if not check_cost_limits():
        return []
    
    # Use cheaper model for trend extraction
    model = "gpt-3.5-turbo" if COST_CONTROLS['use_gpt35_for_trends'] else "gpt-4"
    
    prompt = f"""
Extract 2–5 short trend tags (1–3 words each) summarizing the main themes from the following text.
Return only a JSON list of strings, like ["AI in Finance", "Cloud Migration"].

=== TEXT START ===
{text[:2000]}
=== TEXT END ===
    """
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=COST_CONTROLS['temperature'],
            max_tokens=100  # Trends are short
        )
        
        # Track usage
        track_openai_usage(
            model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            "Trend Extraction"
        )
        
        raw_output = response.choices[0].message.content.strip()
        # Clean up response
        if raw_output.startswith("```"):
            raw_output = raw_output.split("```")[1].strip()
            if raw_output.startswith("json"):
                raw_output = raw_output[4:].strip()
        return json.loads(raw_output)
    except Exception as e:
        return []


def summarize_section(section_name, article_texts, article_links, article_titles, prompts):
    """Summarize a newsletter section using GPT (with cost controls)"""
    if not check_cost_limits():
        return f"(Skipped due to cost limits: {section_name})", []
    
    # Choose model based on config
    model = "gpt-4" if COST_CONTROLS['use_gpt4_for_summaries'] else "gpt-3.5-turbo"
    
    combined_text = "\n\n".join(article_texts)
    # Limit text to control costs
    max_chars = 30000 if model == "gpt-4" else 40000
    if len(combined_text) > max_chars:
        combined_text = combined_text[:max_chars]
    
    sample_citations = "\n".join([f"[{i+1}] {title}" for i, title in enumerate(article_titles[:5])])
    prompt_template = prompts.get(section_name, prompts.get("default", ""))
    prompt = prompt_template.format(
        section_name=section_name,
        today=datetime.now().strftime('%B %d, %Y'),
        combined_text=combined_text,
        sample_citations=sample_citations
    )
    
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=COST_CONTROLS['temperature'],
            max_tokens=COST_CONTROLS['max_tokens_per_request']
        )
        
        # Track usage
        track_openai_usage(
            model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            f"Summary: {section_name[:20]}"
        )
        
        summary_text = response.choices[0].message.content.strip()
        # Bold the labels
        for label in ["Summary:", "Full Brief:", "Key Themes:", "Key Highlights:", "Consulting Relevance:"]:
            summary_text = summary_text.replace(label, f"**{label}**")
        return summary_text, article_links[:5]
    except Exception as e:
        return f"(Error summarizing {section_name}: {e})", []


def generate_academic_summary(papers_by_topic):
    """Generate GPT summary of academic papers (with cost controls)"""
    if not check_cost_limits():
        return "Academic summary skipped due to cost limits."
    
    if not papers_by_topic or not any(papers_by_topic.values()):
        return "No recent academic papers found for this period."
    
    # Compile papers info (limit to save tokens)
    papers_text = ""
    for topic, papers in papers_by_topic.items():
        if papers:
            papers_text += f"\n\n### {topic}:\n"
            for paper in papers[:2]:  # Only top 2 per topic
                papers_text += f"- {paper['title']} ({paper['year']}, {paper['citation_count']} citations)\n"
    
    prompt = f"""
You are an academic research analyst. Summarize the key research themes and findings from recent academic papers.

**Papers:**
{papers_text[:3000]}

**Provide:**
1. **Research Highlights** (2-3 sentences on key developments)
2. **Emerging Themes** (3-5 bullet points on common research directions)
3. **Business Implications** (How these findings relate to consulting and financial services)

Keep it concise and business-focused.
    """
    
    try:
        model = "gpt-3.5-turbo"  # Use cheaper model for academic summary
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=COST_CONTROLS['temperature'],
            max_tokens=800
        )
        
        track_openai_usage(
            model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            "Academic Summary"
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating academic summary: {e}"


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
    
    # Count articles from config
    config_path = os.path.join(base_dir, "config", "e22_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            metrics["articles"] = len(config.get("sources", {})) * 10  # Approximate
    
    # Count trends
    if os.path.exists(trend_csv_path):
        trend_df = pd.read_csv(trend_csv_path)
        today_str = datetime.now().strftime('%Y-%m-%d')
        metrics["trends"] = len(trend_df[trend_df["date"] == today_str]["tag"].unique())
    
    # Market movement (S&P 500 weekly)
    try:
        import yfinance as yf
        sp500 = yf.Ticker("^GSPC")
        hist = sp500.history(period="5d")
        if len(hist) >= 2:
            metrics["market_movement"] = ((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100
    except:
        pass
    
    return metrics


def generate_newsletter():
    """Main newsletter generation function (with cost controls)"""
    # Reset usage tracking for new run
    st.session_state['openai_usage'] = {
        'total_tokens': 0,
        'api_calls': 0,
        'estimated_cost': 0.0,
        'calls_log': []
    }
    
    st.info("📥 Generating this week's newsletter...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load configs
    with open(os.path.join(base_dir, "config", "e22_config.json"), "r") as f:
        config = json.load(f)
    SOURCES = config["sources"]
    
    with open(os.path.join(base_dir, "env.prompts.json"), "r") as f:
        SECTION_PROMPTS = json.load(f)
    
    today = datetime.now().strftime('%B %d, %Y')
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    section_outputs = []
    all_trend_tags = []
    
    # Define section order
    manual_order = [
        "Market & Macro Watch",
        "Financial Services Transformation",
        "AI & Automation in Financial Services",
        "Consulting & Advisory Trends",
        "Innovation & Tech Startups",
        "Data Privacy & Regulatory Compliance",
        "Enterprise Data Management",
        "Policy & Public Sector Data"
    ]
    
    # Process each section
    total_sections = len(manual_order)
    for idx, section in enumerate(manual_order):
        status_text.text(f"🔍 Processing: {section}")
        progress_bar.progress((idx + 1) / (total_sections + 2))
        
        params = SOURCES.get(section, {})
        if not params:
            continue
        
        results = search_tavily(params)
        links = [r["url"] for r in results if r.get("content")]
        texts = [r["content"] for r in results if r.get("content")]
        titles = [r["title"] for r in results if r.get("content")]
        
        if texts:
            summary, used_links = summarize_section(section, texts, links, titles, SECTION_PROMPTS)
            section_outputs.append((section, summary, used_links))
            
            # Extract trends (only for first 4 sections to save costs)
            if idx < 4:
                trends = extract_trends(summary)
                all_trend_tags.extend(trends)
        else:
            section_outputs.append((section, "No updates today.", []))
    
    # Fetch academic papers (optional - can be disabled to save time)
    status_text.text("📚 Fetching academic papers...")
    progress_bar.progress(0.9)
    
    try:
        academic_results = search_academic_papers_by_topics(
            ACADEMIC_TOPICS,
            papers_per_topic=3,  # Reduced from 5 to save time
            days_back=30
        )
    except Exception as e:
        st.warning(f"Academic paper search skipped: {e}")
        academic_results = {}
    
    # Save data
    status_text.text("💾 Saving newsletter data...")
    progress_bar.progress(0.95)
    
    # Save trends
    if all_trend_tags:
        if os.path.exists(trend_csv_path):
            trend_df = pd.read_csv(trend_csv_path)
        else:
            trend_df = pd.DataFrame(columns=["date", "tag", "count"])
        
        tag_counts_today = pd.Series(all_trend_tags).value_counts().to_dict()
        new_entries = pd.DataFrame([
            {"date": today_str, "tag": tag, "count": count}
            for tag, count in tag_counts_today.items()
        ])
        
        trend_df = pd.concat([trend_df, new_entries], ignore_index=True)
        trend_df.to_csv(trend_csv_path, index=False)
    
    # Save raw text
    output_text = f"E22 Weekly Brief – {today}\n\n"
    for section, summary, _ in section_outputs:
        output_text += f"===== {section} =====\n\n{summary}\n\n" + "-"*80 + "\n\n"
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    
    # Store in session state
    st.session_state['section_outputs'] = section_outputs
    st.session_state['academic_results'] = academic_results
    st.session_state['newsletter_generated'] = True
    
    progress_bar.progress(1.0)
    
    # Display final cost
    usage = st.session_state['openai_usage']
    status_text.text(f"✅ Complete! Used {usage['total_tokens']:,} tokens (${usage['estimated_cost']:.4f})")
    st.success(f"✅ Newsletter generated! OpenAI cost: ${usage['estimated_cost']:.4f}")


# === MAIN APP ===

# Initialize session state
if 'newsletter_generated' not in st.session_state:
    st.session_state['newsletter_generated'] = False

# Render hero header
render_hero_header()

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")
nav_options = [
    ("📊 Dashboard", "dashboard"),
    ("🔍 Client Intel", "intel"),
    ("📬 Newsletter", "newsletter"),
    ("📚 Academic Papers", "academic"),
    ("📈 Trends", "trends"),
    ("📁 Archive", "archive")
]

selected_nav = st.sidebar.radio("Go to:", [opt[0] for opt in nav_options])
selected_section = [opt[1] for opt in nav_options if opt[0] == selected_nav][0]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Actions")
if st.sidebar.button("🔄 Regenerate Newsletter"):
    st.session_state['newsletter_generated'] = False
    generate_newsletter()
    st.rerun()

# Display cost dashboard
display_cost_dashboard()

# Generate newsletter if not exists
if not st.session_state['newsletter_generated']:
    generate_newsletter()

# === SECTION: DASHBOARD ===
if selected_section == "dashboard":
    st.markdown("## 📊 Executive Dashboard")
    
    metrics = calculate_kpi_metrics()
    render_kpi_dashboard(metrics)
    
    st.markdown("---")
    
    # Show trend charts
    if os.path.exists(trend_csv_path):
        trend_df = pd.read_csv(trend_csv_path)
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔥 Top Trends This Week")
            week_trends = (
                trend_df[trend_df["date"] == today_str]
                .groupby("tag")["count"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )
            if not week_trends.empty:
                fig = create_trend_chart(week_trends, "This Week's Trends", "#1f77b4")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Top Trends Overall")
            overall_trends = (
                trend_df.groupby("tag")["count"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )
            if not overall_trends.empty:
                fig = create_trend_chart(overall_trends, "Overall Trends", "#2ca02c")
                st.plotly_chart(fig, use_container_width=True)

# === SECTION: CLIENT INTEL ===
elif selected_section == "intel":
    st.markdown("## 🔍 Client Intel Search")
    
    st.markdown("""
    <div class="info-box info">
        <div class="info-box-title">💡 Search Tips</div>
        Enter a company name to get recent strategic developments, AI initiatives, and consulting opportunities.
    </div>
    """, unsafe_allow_html=True)
    
    company = st.text_input("Enter company name:", placeholder="e.g., BlackRock, JPMorgan, Goldman Sachs")
    
    if company:
        with st.spinner(f"🔎 Researching {company}..."):
            try:
                results = search_tavily({
                    "query": f"{company} strategy performance digital transformation AI",
                    "max_results": 10,
                    "time_range": "week"
                })
                
                if results and check_cost_limits():
                    full_text = "\n\n".join([r["content"] for r in results if r.get("content")])
                    link_list = "\n".join([r["url"] for r in results if r.get("url")])
                    
                    gpt_prompt = f"""
You are an industry analyst. Analyze recent developments at {company}.

**Structure:**
**Summary:**
(2-3 sentences on key developments)

**Strategic Initiatives:**
- Recent AI/data projects
- Digital transformation efforts
- Technology investments

**Consulting Opportunities:**
- Where could Element22 add value?
- What capabilities do they need?

**Sources:**
{link_list}

=== CONTENT ===
{full_text[:6000]}
                    """
                    
                    model = "gpt-3.5-turbo"  # Use cheaper model for client intel
                    response = openai.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": gpt_prompt}],
                        temperature=0.5,
                        max_tokens=1000
                    )
                    
                    track_openai_usage(
                        model,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                        f"Client Intel: {company}"
                    )
                    
                    summary = response.choices[0].message.content.strip()
                    render_section_card(
                        f"Intel Brief: {company}",
                        summary.replace("\n", "<br>"),
                        "🎯"
                    )
                else:
                    st.info(f"No recent results found for {company}.")
            except Exception as e:
                st.error(f"Error: {e}")

# === SECTION: NEWSLETTER ===
elif selected_section == "newsletter":
    st.markdown("## 📬 This Week's Newsletter")
    
    if 'section_outputs' in st.session_state:
        section_icons = {
            "Market & Macro Watch": "📊",
            "Financial Services Transformation": "🏦",
            "AI & Automation in Financial Services": "🤖",
            "Consulting & Advisory Trends": "💼",
            "Innovation & Tech Startups": "🚀",
            "Data Privacy & Regulatory Compliance": "🔒",
            "Enterprise Data Management": "💾",
            "Policy & Public Sector Data": "🏛️"
        }
        
        # Show market chart first
        st.markdown("### 📈 Market Performance")
        try:
            fig = create_market_performance_chart()
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load market chart: {e}")
        
        st.markdown("---")
        
        # Show all sections
        for section, summary, references in st.session_state['section_outputs']:
            icon = section_icons.get(section, "📄")
            render_section_card(section, summary.replace("\n", "<br>"), icon, references)

# === SECTION: ACADEMIC PAPERS ===
elif selected_section == "academic":
    st.markdown("## 📚 Academic Insights")
    
    if 'academic_results' in st.session_state:
        papers_by_topic = st.session_state['academic_results']
        
        # Generate summary
        with st.spinner("Analyzing research papers..."):
            summary = generate_academic_summary(papers_by_topic)
            st.markdown(f"""
            <div class="section-card">
                <div class="section-header">
                    <span class="section-icon">🧠</span>
                    <h2 class="section-title">Research Highlights</h2>
                </div>
                <div class="section-content">
                    {summary.replace(chr(10), "<br>")}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Show papers by topic
        for topic, papers in papers_by_topic.items():
            if papers:
                st.markdown(f"### {topic}")
                st.markdown(f"*{len(papers)} recent papers*")
                
                for paper in papers:
                    render_academic_paper_card(paper)
                
                st.markdown("---")

# === SECTION: TRENDS ===
elif selected_section == "trends":
    st.markdown("## 📈 Trend Analysis")
    
    if os.path.exists(trend_csv_path):
        trend_df = pd.read_csv(trend_csv_path)
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        # Comparison chart
        st.markdown("### Trend Comparison")
        fig = create_trend_comparison_chart(trend_df, today_str)
        st.plotly_chart(fig, use_container_width=True)

# === SECTION: ARCHIVE ===
elif selected_section == "archive":
    st.markdown("## 📁 Newsletter Archive")
    
    if os.path.exists(html_dir):
        html_files = sorted(
            [f for f in os.listdir(html_dir) if f.endswith(".html")],
            reverse=True
        )
        
        st.markdown(f"**{len(html_files)} past issues available**")
        
        for html_file in html_files[:10]:
            if html_file == f"{year}-W{week_num}.html":
                continue
            
            week_label = html_file.replace(".html", "")
            with st.expander(f"📅 Week {week_label}"):
                file_path = os.path.join(html_dir, html_file)
                with open(file_path, "r", encoding="utf-8") as f:
                    st.markdown(f.read(), unsafe_allow_html=True)

# Back to top button
st.markdown("""
<div class="back-to-top" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">
    ↑
</div>
""", unsafe_allow_html=True)