import streamlit as st
import os
import json
import requests
from datetime import datetime
import openai
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# === FIRST STREAMLIT COMMAND ===
st.set_page_config(page_title="E22 Weekly Brief", layout="wide")

# === SETUP ===
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Debugging aid (optional: remove later)
if not TAVILY_API_KEY:
    st.error("❌ Tavily API key not found in environment variables. Check GitHub Secrets.")
else:
    st.success(f"✅ Tavily API key loaded successfully: `{TAVILY_API_KEY`")

    
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

# === FUNCTIONS ===

def generate_newsletter():
    st.info("📥 Generating this week's newsletter...")

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

    # Helper functions inside
    def search_tavily(params):
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
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            st.error(f"❌ Tavily query failed: {e}")
            return []

    def extract_trends(text):
        prompt = f"""
Extract 2–5 short trend tags (1–3 words each) summarizing the main themes from the following text.
Return only a JSON list of strings, like ["AI in Finance", "Cloud Migration"].

=== TEXT START ===
{text}
=== TEXT END ===
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            raw_output = response.choices[0].message.content.strip()
            try:
                return json.loads(raw_output)
            except json.JSONDecodeError:
                return []
        except Exception as e:
            return []

    def summarize_section(section_name, article_texts, article_links, article_titles, today):
        combined_text = "\n\n".join(article_texts)
        if len(combined_text) > 48000:
            combined_text = combined_text[:48000]

        sample_citations = "\n".join([f"- {title} ({link})" for title, link in zip(article_titles[:3], article_links[:3])])
        prompt_template = SECTION_PROMPTS.get(section_name, SECTION_PROMPTS.get("default", ""))
        prompt = prompt_template.format(
            section_name=section_name,
            today=today,
            combined_text=combined_text,
            sample_citations=sample_citations
        )

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"(Error summarizing {section_name}: {e})"

    for section, params in SOURCES.items():
        query = params.get("query", "")
        st.write(f"🔍 Fetching: {section}")
        results = search_tavily(params)

        links = [r["url"] for r in results if r.get("content")]
        texts = [r["content"] for r in results if r.get("content")]
        titles = [r["title"] for r in results if r.get("content")]

        if texts:
            summary = summarize_section(section, texts, links, titles, today)
            section_outputs.append((section, summary))
            # Trend tag extraction
            trends = extract_trends(summary)
            all_trend_tags.extend(trends)
        else:
            section_outputs.append((section, "No updates today."))

    # Save raw text
    output_text = f"E22 Weekly Brief – {today}\n\n"
    for section, summary in section_outputs:
        output_text += f"===== {section} =====\n\n{summary}\n\n" + "-"*80 + "\n\n"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    # Save trends
    if all_trend_tags:
        today_str = datetime.now().strftime('%Y-%m-%d')
        if os.path.exists(trend_csv_path):
            trend_df = pd.read_csv(trend_csv_path)
        else:
            trend_df = pd.DataFrame(columns=["date", "tag", "count"])

        tag_counts_today = pd.Series(all_trend_tags).value_counts().to_dict()
        new_entries = pd.DataFrame([
            {"date": today_str, "tag": tag, "count": count} for tag, count in tag_counts_today.items()
        ])

        trend_df = pd.concat([trend_df, new_entries], ignore_index=True)
        trend_df.to_csv(trend_csv_path, index=False)

    # Build newsletter HTML
    final_output_html = f"""
    <html>
    <head>
      <style>
        body {{ background-color: white; font-family: Arial, sans-serif; line-height: 1.6; color: #222; }}
        h1 {{ color: #0056b3; }}
        h2 {{ color: #444; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
        .section {{ margin-bottom: 30px; }}
        .links {{ margin-top: 10px; font-size: 0.95em; color: #555; }}
      </style>
    </head>
    <body>
      <h1>🗞️ E22 Weekly Brief – {today}</h1>
      <p><em>Internal newsletter summarizing recent developments across key areas.</em></p>
      <hr>
    """

    # Embed charts
    if os.path.exists(trend_csv_path):
        trend_df = pd.read_csv(trend_csv_path)

        top_today = (
            trend_df[trend_df["date"] == today_str]
            .groupby("tag")["count"].sum()
            .sort_values(ascending=False)
            .head(5)
        )

        top_overall = (
            trend_df.groupby("tag")["count"].sum()
            .sort_values(ascending=False)
            .head(5)
        )

        def plot_to_base64(series, title, color):
            fig, ax = plt.subplots(figsize=(6, 4))
            series.plot(kind="bar", ax=ax, color=color)
            ax.set_title(title)
            ax.set_ylabel("Mentions")
            ax.set_xticklabels(series.index, rotation=45, ha="right")
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            return f'<img src="data:image/png;base64,{img_base64}" alt="{title}"/>'

        final_output_html += f"""
        <div class="section">
          <h2>🔥 Top Trends This Week</h2>
          {plot_to_base64(top_today, "Top Trends This Week", "#1f77b4")}
          <h2>📈 Top Trends Overall</h2>
          {plot_to_base64(top_overall, "Top Trends Overall", "#2ca02c")}
        </div>
        """

    for section, summary in section_outputs:
        summary_html = summary.replace("\n", "<br>")
        final_output_html += f"""
        <div class="section">
          <h2>{section}</h2>
          <p>{summary_html}</p>
        </div>
        """

    final_output_html += """
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_output_html)

    st.success("✅ Newsletter generated, saved, and charts embedded!")

# === MAIN EXECUTION ===

st.title("🗞️ E22 Weekly Newsletter")

st.subheader("🔍 Client Intel Search")

company = st.text_input("Enter a company name to get recent news:", placeholder="e.g. BlackRock, Vanguard")

if company:
    with st.spinner("Searching recent news about the company..."):
        try:
            url = "https://api.tavily.com/search"
            headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
            query = f"{company} strategy OR performance OR digital transformation"

            payload = {
                "query": query,
                "search_depth": "basic",  # simplified for reliability
                "time_range": "week",
                "max_results": 10,
                "include_content": True,
                "include_titles": True
            }

            response = requests.post(url, json=payload, headers=headers, timeout=20)
            response.raise_for_status()
            results = response.json().get("results", [])

        except requests.exceptions.HTTPError as e:
            error_msg = e.response.text if e.response else str(e)
            st.error(f"❌ Tavily query failed ({e.response.status_code}): {error_msg}")
            results = []
        except Exception as e:
            st.error(f"❌ Unexpected error fetching Tavily results: {e}")
            results = []

    if results:
        full_text = "\n\n".join([r["content"] for r in results if r.get("content")])
        link_list = "\n".join([r["url"] for r in results if r.get("url")])

        gpt_prompt = f"""
You are an industry analyst summarizing the latest strategic developments at {company}. Use only the content provided below.

Summarize key developments, AI and data initiatives, leadership strategy, and consulting relevance for a firm like Element22. Respond in this format:

Summary:
(Concise 1–2 sentence summary)

Key Highlights:
- (Insight 1)
- (Insight 2)
- (Etc.)

Consulting Relevance:
- Why this is important for data consultants
- Questions we might ask this client

Links:
{link_list}

=== CONTENT START ===
{full_text}
=== CONTENT END ===
        """

        try:
            summary_response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": gpt_prompt}]
            )
            summary_output = summary_response.choices[0].message.content.strip()
            st.markdown("### 🧠 GPT Summary")
            st.markdown(summary_output)
        except Exception as e:
            st.error(f"❌ GPT summarization failed: {e}")
    else:
        st.info(f"No recent results found for {company}.")

if not os.path.exists(html_path):
    generate_newsletter()

st.markdown("---")
st.subheader("📊 Feedback Poll")

st.markdown(
    """
    <iframe src="https://docs.google.com/forms/d/e/1FAIpQLSe3jz1gBRprUXjDRUjG0NiGTTLqwNWi8oIO4z153zDkNsWTJA/viewform?embedded=true" 
        width="1800" height="500" frameborder="0" marginheight="0" marginwidth="0">
    Loading…
    </iframe>
    """,
    unsafe_allow_html=True
)

st.subheader("📬 This Week's Newsletter")

with open(html_path, "r", encoding="utf-8") as f:
    newsletter_html = f.read()

st.components.v1.html(
    f"""
    <div style="background-color: white; padding: 20px;">
        {newsletter_html}
    </div>
    """,
    height=1600,
    scrolling=True
)
