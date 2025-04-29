import streamlit as st
import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
import openai
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# === FIRST STREAMLIT COMMAND ===
st.set_page_config(page_title="E22 Weekly Brief", layout="wide")

# === SETUP ===
load_dotenv("env")
openai.api_key = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

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
                model="gpt-3.5-turbo",
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
                model="gpt-3.5-turbo",
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

    for section, summary in section_outputs:
        summary_html = summary.replace("\n", "<br>")
        final_output_html += f"""
        <div class="section">
          <h2>{section}</h2>
          <p>{summary_html}</p>
        </div>
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

    final_output_html += """
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_output_html)

    st.success("✅ Newsletter generated, saved, and charts embedded!")

# === MAIN EXECUTION ===

st.title("🗞️ E22 Weekly Newsletter")

if not os.path.exists(html_path):
    generate_newsletter()

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