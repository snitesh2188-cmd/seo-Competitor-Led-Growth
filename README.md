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

Python
import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
import datetime
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Growth Matrix", layout="wide")

st.title("🚀 AI Competitor Growth Matrix")
st.markdown("Upload Ahrefs competitor data, merge it with GSC, and use Semantic AI to uncover targeted growth clusters.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. API Settings")
property_uri = st.sidebar.text_input("GSC Property", value="sc-domain:example.com", help="e.g. sc-domain:example.com or https://example.com/")

country_map = {"Worldwide (No Filter)": None, "Singapore": "sgp", "Australia": "aus", "United States": "usa", "United Kingdom": "gbr", "India": "ind"}
selected_country = st.sidebar.selectbox("Target Country", options=list(country_map.keys()))

st.sidebar.header("2. Analysis Filters")
time_option = st.sidebar.selectbox("Date Range", ("Last 28 Days", "Last 3 Months", "Last 6 Months", "Last 12 Months"))
branded_input = st.sidebar.text_area("Your Brand Terms (comma separated)", value="brandname")

# Date Logic
today = datetime.date.today()
delta = {"Last 28 Days": 28, "Last 3 Months": 90, "Last 6 Months": 180, "Last 12 Months": 365}[time_option]
start_date_str = (today - datetime.timedelta(days=delta)).strftime("%Y-%m-%d")
end_date_str = today.strftime("%Y-%m-%d")

# --- CORE FUNCTIONS ---
def fetch_gsc_data(site_url, start, end, country_code=None, keys_path='credentials.json', limit=10000):
    try:
        creds = service_account.Credentials.from_service_account_file(keys_path, scopes=['https://www.googleapis.com/auth/webmasters.readonly'])
        service = build('searchconsole', 'v1', credentials=creds)
        request_body = {'startDate': start, 'endDate': end, 'dimensions': ['query'], 'rowLimit': limit, 'startRow': 0}
        if country_code: request_body['dimensionFilterGroups'] = [{'filters': [{'dimension': 'country', 'operator': 'equals', 'expression': country_code}]}]
        response = service.searchanalytics().query(siteUrl=site_url, body=request_body).execute()
        if 'rows' not in response: return None
        return pd.DataFrame([{'query': r['keys'][0], 'clicks': r['clicks'], 'impressions': r['impressions'], 'position': r['position']} for r in response['rows']])
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# --- UI: FILE UPLOADERS ---
col1, col2 = st.columns(2)
comp1_file = col1.file_uploader("Competitor 1 Ahrefs Export (CSV)", type=['csv'])
comp2_file = col2.file_uploader("Competitor 2 Ahrefs Export (CSV)", type=['csv'])

# --- PHASE 1: DATA PROCESSING ENGINE ---
if st.button("Generate Growth Matrix", type="primary"):
    if not comp1_file and not comp2_file:
        st.warning("Please upload at least one competitor file.")
    else:
        with st.spinner("Crunching mega-pool data and clustering intents with Semantic AI (This may take a minute)..."):
            master_data = []
            
            # 1. Fetch GSC Data
            if not os.path.exists('credentials.json'):
                st.error("❌ 'credentials.json' file not found! Please create one from Google Cloud Console.")
                st.stop()
                
            gsc_df = fetch_gsc_data(property_uri, start_date_str, end_date_str, country_code=country_map[selected_country])
            
            if gsc_df is not None and not gsc_df.empty:
                if branded_input:
                    terms = [t.strip().lower() for t in branded_input.split(',') if t.strip()]
                    pattern = '|'.join(terms)
                    if pattern: gsc_df = gsc_df[~gsc_df['query'].str.contains(pattern, case=False, regex=True)]
                        
                gsc_df = gsc_df.sort_values('impressions', ascending=False).head(5000)
                gsc_best_pos = gsc_df.groupby('query')['position'].min().to_dict()
                gsc_clicks = gsc_df.groupby('query')['clicks'].sum().to_dict()
                
                for kw, pos in gsc_best_pos.items():
                    master_data.append({"Keyword": str(kw).strip(), "Position": pos, "Traffic": gsc_clicks.get(kw, 0), "Source": "GSC (You)"})
            
            # 2. Parse Ahrefs Data
            def process_ahrefs(file, source_name):
                try:
                    df = pd.DataFrame()
                    strategies = [{'encoding': 'utf-8', 'sep': ','}, {'encoding': 'utf-16', 'sep': '\t'}, {'encoding': 'latin1', 'sep': ','}]
                    for strat in strategies:
                        try:
                            file.seek(0)
                            df = pd.read_csv(file, encoding=strat['encoding'], sep=strat['sep'], on_bad_lines='skip', engine='python')
                            if len(df.columns) > 2: break
                        except: continue
                            
                    kw_col = next((col for col in df.columns if 'keyword' in col.lower()), None)
                    trf_col = next((col for col in df.columns if 'traffic' in col.lower()), None)
                    pos_col = next((col for col in df.columns if 'position' in col.lower()), None)
                    
                    if kw_col and trf_col and pos_col:
                        df[kw_col] = df[kw_col].astype(str).fillna('')
                        df = df[df[kw_col] != '']
                        df = df.sort_values(trf_col, ascending=False).head(1000)
                        
                        for _, row in df.iterrows():
                            master_data.append({
                                "Keyword": str(row[kw_col]).strip(),
                                "Position": float(row[pos_col]) if pd.notna(row[pos_col]) else 999.0,
                                "Traffic": float(row[trf_col]) if pd.notna(row[trf_col]) else 0.0,
                                "Source": source_name
                            })
                except Exception as e: st.error(f"Error reading {source_name}: {e}")

            if comp1_file: process_ahrefs(comp1_file, "Competitor 1")
            if comp2_file: process_ahrefs(comp2_file, "Competitor 2")
            
            master_df = pd.DataFrame(master_data)
            
            if master_df.empty or len(master_df) < 2:
                st.error("❌ Not enough valid keywords found. Check your files.")
            else:
                # 3. True Semantic AI Clustering (BERT)
                master_df['Keyword'] = master_df['Keyword'].astype(str).fillna('')
                
                model = SentenceTransformer('all-MiniLM-L6-v2')
                X_dense = model.encode(master_df['Keyword'].tolist())
                
                clustering = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.4)
                master_df['Cluster_ID'] = clustering.fit_predict(X_dense)
                
                # Name cluster by HIGHEST COMPETITOR TRAFFIC keyword
                cluster_names = master_df.sort_values('Traffic', ascending=False).groupby('Cluster_ID')['Keyword'].first().to_dict()
                master_df['Cluster_Name'] = master_df['Cluster_ID'].map(cluster_names)
                
                # 4. Roll-Up & Logic
                gsc_mask = master_df['Source'] == 'GSC (You)'
                comp_mask = master_df['Source'] != 'GSC (You)'
                
                gsc_best = master_df[gsc_mask].groupby('Cluster_Name')['Position'].min()
                comp_best = master_df[comp_mask].groupby('Cluster_Name')['Position'].min()
                comp_traffic = master_df[comp_mask].groupby('Cluster_Name')['Traffic'].sum()
                
                cluster_df = pd.DataFrame({'Best_GSC_Position': gsc_best, 'Best_Comp_Position': comp_best, 'Opportunity_Traffic': comp_traffic}).reset_index()
                cluster_df = cluster_df[cluster_df['Opportunity_Traffic'] > 0]
                cluster_df['Opportunity_Traffic'] = cluster_df['Opportunity_Traffic'].fillna(0).astype(int)
                
                def apply_matrix_logic(row):
                    gsc = row['Best_GSC_Position']
                    comp = row['Best_Comp_Position']
                    if pd.isna(gsc): return "🔴 Red (Content Gap)"
                    if pd.notna(comp) and gsc < comp: return "🟢 Green (Winning)"
                    if gsc <= 20: return "🔵 Blue (Striking Distance)"
                    return "🟡 Yellow (Needs Work)"

                cluster_df['Status'] = cluster_df.apply(apply_matrix_logic, axis=1)
                
                # Save to session memory
                st.session_state['cluster_df'] = cluster_df
                st.session_state['master_df'] = master_df
                st.success("✅ Analysis Complete! Scroll down to explore.")

# --- PHASE 2: RENDERING ENGINE ---
if 'cluster_df' in st.session_state and 'master_df' in st.session_state:
    cluster_df = st.session_state['cluster_df']
    master_df = st.session_state['master_df']
    
    color_discrete_map = {"🔴 Red (Content Gap)": "#FF4B4B", "🟢 Green (Winning)": "#00CC96", "🟡 Yellow (Needs Work)": "#FECB52", "🔵 Blue (Striking Distance)": "#636EFA"}

    # 1. Data Table
    st.subheader("📊 Competitor-Led Growth Clusters")
    st.dataframe(
        cluster_df.sort_values('Opportunity_Traffic', ascending=False),
        use_container_width=True,
        column_config={
            "Best_GSC_Position": st.column_config.NumberColumn("Our Best Pos.", format="%.1f"),
            "Best_Comp_Position": st.column_config.NumberColumn("Comp Best Pos.", format="%.1f"),
            "Opportunity_Traffic": st.column_config.NumberColumn("Comp Traffic Vol.", format="%d")
        }, hide_index=True
    )

    # 2. Treemap
    st.divider()
    st.subheader("🗺️ Growth Cluster Treemap")
    treemap_df = cluster_df.copy()
    treemap_df['Root'] = 'All Growth Opportunities'
    treemap_df = treemap_df.fillna("Unknown")
    
    fig_tree = px.treemap(treemap_df, path=['Root', 'Status', 'Cluster_Name'], values='Opportunity_Traffic', color='Status', color_discrete_map=color_discrete_map)
    fig_tree.update_layout(margin = dict(t=20, l=0, r=0, b=0))
    st.plotly_chart(fig_tree, use_container_width=True)

    # 3. Forecast Chart
    st.divider()
    st.subheader("📈 12-Month Traffic Forecast (2% MoM Growth)")
    forecast_data = []
    for _, row in cluster_df.groupby('Status')['Opportunity_Traffic'].sum().reset_index().iterrows():
        base = row['Opportunity_Traffic']
        for month in range(13): forecast_data.append({"Status": row['Status'], "Month": month, "Projected Traffic": int(base * ((1.02) ** month))})
            
    fig_area = px.area(pd.DataFrame(forecast_data), x='Month', y='Projected Traffic', color='Status', color_discrete_map=color_discrete_map, markers=True)
    fig_area.update_layout(xaxis_title="Months from Now", yaxis_title="Estimated Cumulative Traffic")
    st.plotly_chart(fig_area, use_container_width=True)

    # 4. Drill Down
    st.divider()
    st.subheader("🔍 Cluster Drill-Down")
    selected_cluster = st.selectbox("Select a Cluster to Inspect:", options=cluster_df.sort_values('Opportunity_Traffic', ascending=False)['Cluster_Name'].tolist())
    
    if selected_cluster:
        drilldown_df = master_df[master_df['Cluster_Name'] == selected_cluster].copy()
        drilldown_df = drilldown_df[['Keyword', 'Source', 'Position', 'Traffic']].sort_values(by=['Source', 'Traffic'], ascending=[True, False])
        drilldown_df['Position'] = drilldown_df['Position'].apply(lambda x: "Not Ranking" if x == 999.0 else f"{x:.1f}")
        drilldown_df['Traffic'] = drilldown_df['Traffic'].astype(int)
        st.dataframe(drilldown_df, use_container_width=True, hide_index=True)
File 2: requirements.txt
(This includes all the required libraries to run the script).

Plaintext
streamlit
pandas
google-auth
google-api-python-client
scikit-learn
sentence-transformers
plotly
numpy
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

### 2. Set Up Google Cloud Credentials
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
