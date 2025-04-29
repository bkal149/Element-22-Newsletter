# 🗞️ E22 Weekly Newsletter App

An automated, AI-powered newsletter generator and viewer built with **Streamlit**, **OpenAI GPT**, and **Tavily** search APIs.  
Generates a custom internal weekly newsletter with:
- GPT-powered section summaries
- Trend extraction and analysis
- Embedded top trend charts
- Clean HTML output for browser viewing

---

## 🚀 Features

- 📥 Auto-generates a full newsletter if none exists for the week
- 📚 Summarizes real-time articles across selected sections
- 🧠 Extracts key takeaways and trending topics
- 📈 Tracks trends week-over-week and all-time
- 🖼️ Embeds trend charts directly in the newsletter
- 🎨 Streamlit app for easy browser viewing (white background styling)

---

## 🧩 Project Structure

```plaintext
newsletter-project/
├── final_newsletter.py         # Main Streamlit app (all logic inside)
├── config/
│   └── e22_config.json         # Newsletter section search queries
├── env.prompts.json            # Prompt templates for GPT summarization
├── requirements.txt            # Dependencies list
├── .gitignore                  # Ignores generated and sensitive files
(newsletter/html/)              # (Generated) Weekly HTML newsletters
(newsletter/raw/)               # (Generated) Weekly raw text backups
(newsletter/trends/)            # (Generated) Trend tracking CSVs
