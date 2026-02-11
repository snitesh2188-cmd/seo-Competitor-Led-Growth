# 🚀 AI-Powered Competitor Growth Matrix

**Transform competitor keyword data into an actionable, intent-based editorial calendar.**

Most SEO scripts match keywords exactly. If a competitor ranks for "SEO agencies" and you rank for "SEO agency," traditional scripts falsely report a content gap. 

This tool uses **Semantic AI (`sentence-transformers` & BERT)** running locally on your machine to understand the *meaning* of keywords. It merges your Google Search Console data with Ahrefs competitor exports to build a strategic Growth Matrix.

## ✨ Features
* **🧠 True Intent Clustering:** Groups thousands of keywords mathematically based on meaning, not just letters. 
* **🗺️ Interactive Treemap:** Visualizes exactly where the competitor traffic is hiding.
* **🚦 Actionability Matrix:** Automatically categorizes topics into Red (Content Gap), Green (Winning), Blue (Striking Distance), and Yellow (Needs Work).
* **📈 Forecasting Engine:** Generates a 12-month traffic projection assuming a 2% MoM compound growth rate.
* **🔍 X-Ray Drill Down:** Click any cluster to see the exact keywords, sources, and ranking disparities inside it.

---

## ⚙️ Installation Guide

### 1. Download the Code
Clone this repository or download the ZIP file.

### 2. Install Dependencies
Open your terminal, navigate to the folder, and run:

bash
pip install -r requirements.txt

3. Set Up Google Cloud Credentials
Go to the Google Cloud Console.

Create a Project and Enable "Google Search Console API".

Create a Service Account and download the JSON key.

Rename the key to credentials.json and place it in the same folder as dashboard.py.

Add the bot's email address as an Owner or Full User in your Google Search Console property settings.

Gemini said
I cannot generate a direct clickable download link for files right here in the chat, but I can give you the exact, complete code blocks for all three files.

All you need to do is open VS Code, create these three empty files in a new folder, and copy-paste the contents below.

File 1: dashboard.py
(This is the complete, standalone AI Growth Matrix tool).

File 2: requirements.txt
(This includes all the required libraries to run the script).

File 3: README.md
(Ready for GitHub).

Markdown
# 🚀 AI-Powered Competitor Growth Matrix

**Transform competitor keyword data into an actionable, intent-based editorial calendar.**

Most SEO scripts match keywords exactly. If a competitor ranks for "SEO agencies" and you rank for "SEO agency," traditional scripts falsely report a content gap. 

This tool uses **Semantic AI (`sentence-transformers` & BERT)** running locally on your machine to understand the *meaning* of keywords. It merges your Google Search Console data with Ahrefs competitor exports to build a strategic Growth Matrix.

## ✨ Features
* **🧠 True Intent Clustering:** Groups thousands of keywords mathematically based on meaning, not just letters. 
* **🗺️ Interactive Treemap:** Visualizes exactly where the competitor traffic is hiding.
* **🚦 Actionability Matrix:** Automatically categorizes topics into Red (Content Gap), Green (Winning), Blue (Striking Distance), and Yellow (Needs Work).
* **📈 Forecasting Engine:** Generates a 12-month traffic projection assuming a 2% MoM compound growth rate.
* **🔍 X-Ray Drill Down:** Click any cluster to see the exact keywords, sources, and ranking disparities inside it.

---

## ⚙️ Installation Guide

### 1. Download the Code
Clone this repository or download the ZIP file.

### 2. Install Dependencies
Open your terminal, navigate to the folder, and run:
```bash
pip install -r requirements.txt
(Note: The first time you run this app, it will download a ~80MB open-source AI model in the background. It will run instantly on subsequent uses).

### 3. Set Up Google Cloud Credentials
* **Go to the Google Cloud Console.
* **Create a Project and Enable "Google Search Console API".
* **Create a Service Account and download the JSON key.
* **Rename the key to credentials.json and place it in the same folder as dashboard.py.
* **Add the bot's email address as an Owner or Full User in your Google Search Console property settings.

🚀 How to Run
* **Open your terminal inside the project folder.
* **Run the Streamlit command:
streamlit run dashboard.py
* **Enter your GSC property in the sidebar.
* **Export your competitor's Organic Keywords from Ahrefs (CSV format). Tip: Filter out the competitor's brand name inside Ahrefs before exporting.
* **Drag and drop the CSVs into the dashboard and click Generate.
