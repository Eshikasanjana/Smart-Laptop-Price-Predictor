<<<<<<< HEAD
=======

>>>>>>> 912b6f5 (Update logic to validate output against API data)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit.components.v1 as components
<<<<<<< HEAD
=======
import requests
from dotenv import load_dotenv
import os

# Force dotenv to override anything else
load_dotenv(dotenv_path=".env", override=True)

API_KEY = os.getenv("SERPER_API_KEY")

def get_live_price(query):
    if not API_KEY:
        return None

    url = "https://google.serper.dev/shopping"
    payload = {"q": query}
    headers = {
        'X-API-KEY': API_KEY,
        'Content-Type': 'application/json'
    }
    try:
        # 1. Send the request
        print("Sending request..." )
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        response.raise_for_status() # Check if the API key is valid/working
        data = response.json()
        print(f"API Response for '{query}': {data}") # Debug log to see the raw response
        
        # 2. Extract price from results
        prices = []
        if 'shopping' in data:
            for item in data['shopping']:
                # Clean "$1,200.00" -> 1200.0
                price_str = item.get('price', '').replace('$', '').replace(',', '')
                if price_str:
                    try:
                        prices.append(float(price_str))
                    except ValueError:
                        continue
        
        # 3. Return the lowest market price found
        return min(prices) if prices else None

    except requests.exceptions.RequestException as e:
        # This catches internet issues, timeouts, or API errors
        print(f"Connection error: {e}")
        return None
>>>>>>> 912b6f5 (Update logic to validate output against API data)

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

# ------------------ PATHS ------------------
BASE_PATH = "/Users/eshikasanjana/Documents/Smart-Laptop-Price-Predictor"
PIPE_PATH = os.path.join(BASE_PATH, "pipe.pkl")
TRAINED_CSV = os.path.join(BASE_PATH, "traineddata_updated.csv")
USD_RATE = 87.0

# ------------------ HELPERS ------------------
def safe_load_pipeline(path):
    if not os.path.exists(path):
        st.error(f"pipe.pkl missing at: {path}")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)

def find_ct(pipe):
    if isinstance(pipe, ColumnTransformer): return pipe
    if isinstance(pipe, Pipeline):
        for s in pipe.named_steps.values():
            if isinstance(s, ColumnTransformer): return s
            if isinstance(s, Pipeline):
                for s2 in s.named_steps.values():
                    if isinstance(s2, ColumnTransformer): return s2
    return None

def safe_unique(df, col, fallback):
    if df is not None and col in df.columns:
        v=df[col].dropna().unique().tolist()
        return v if len(v)>0 else fallback
    return fallback

def safe_median(df, col, default):
    if df is not None and col in df.columns:
        x=df[col].dropna()
        if len(x)>0:
            try: return float(x.median())
            except: pass
    return default

# ------------------ LOAD MODEL & CSV ------------------
rf = safe_load_pipeline(PIPE_PATH)

train_df=None
if os.path.exists(TRAINED_CSV):
    try: train_df = pd.read_csv(TRAINED_CSV)
    except: train_df=None

ct = find_ct(rf)
if ct and hasattr(ct,"feature_names_in_"):
    expected_cols=list(ct.feature_names_in_)
elif hasattr(rf,"feature_names_in_"):
    expected_cols=list(rf.feature_names_in_)
elif train_df is not None:
    expected_cols=list(train_df.columns)
else:
    st.error("Cannot determine model input feature order.")
    st.stop()

# ------------------ STREAMLIT SIDEBAR INPUTS ------------------
st.sidebar.header("Laptop Specs")

company = st.sidebar.selectbox("Brand", safe_unique(train_df,"Company",["Dell","HP","Apple","Lenovo"]))

if company=="Apple":
    t = safe_unique(train_df,"TypeName",[])
    t=[x for x in t if "mac" in str(x).lower()] or ["MacBook Air","MacBook Pro"]
else:
    t = safe_unique(train_df,"TypeName",["Notebook","Ultrabook","Gaming"])
type_name = st.sidebar.selectbox("Type", t)

cpu = st.sidebar.selectbox("CPU", safe_unique(train_df,"CPU_name",["Intel Core i5","Intel Core i7"]))

screen_size = st.sidebar.number_input(
    "Screen Size", value=safe_median(train_df,"Inches",15.6), step=0.1
)
resolution = st.sidebar.selectbox(
    "Resolution", ['1920x1080','1366x768','1600x900','2560x1440','3840x2160']
)
weight = st.sidebar.number_input(
    "Weight (kg)", value=safe_median(train_df,"Weight",1.5), step=0.1
)
ram = st.sidebar.selectbox("RAM (GB)", safe_unique(train_df,"Ram",[4,8,16,32]))
gpu = st.sidebar.selectbox("GPU", safe_unique(train_df,"Gpu brand",["Intel","Nvidia","AMD"]))
touch = st.sidebar.selectbox("Touchscreen",["No","Yes"])
ips = st.sidebar.selectbox("IPS",["No","Yes"])

if company=="Apple":
    hdd=0
    ssd=st.sidebar.selectbox("SSD (GB)", safe_unique(train_df,"SSD",[128,256,512,1024]))
    os_name = st.sidebar.selectbox("OS",["MacOS"],disabled=True)
else:
    hdd=st.sidebar.selectbox("HDD (GB)", safe_unique(train_df,"HDD",[0,128,256,512,1024]))
    ssd=st.sidebar.selectbox("SSD (GB)", safe_unique(train_df,"SSD",[0,128,256,512,1024]))
    os_name=st.sidebar.selectbox("OS", safe_unique(train_df,"OpSys",["Windows","Linux","Other"]))

# Keep the functional sidebar button that triggers prediction
predict_clicked = st.sidebar.button("Predict Price")

# ------------------ BUILD MODEL INPUT ROW ------------------
# PPI
try:
    X,Y = map(int,resolution.split("x"))
    ppi = ((X**2 + Y**2)**0.5) / screen_size
except:
    ppi = 140

candidate_map = {
    "Company":company, "TypeName":type_name, "Ram":ram, "OpSys":os_name,
    "Weight":weight, "TouchScreen":1 if touch=="Yes" else 0,
    "Touchscreen":1 if touch=="Yes" else 0,
    "IPS":1 if ips=="Yes" else 0, "PPI":ppi,
    "CPU_name":cpu, "CPU":cpu, "HDD":hdd, "SSD":ssd,
    "Gpu brand":gpu, "Gpu":gpu, "GpuBrand":gpu
}

row=[]
for col in expected_cols:
    if col in candidate_map:
        row.append(candidate_map[col])
    else:
        if train_df is not None and col in train_df.columns:
            m=train_df[col].dropna()
            row.append(m.mode().iloc[0] if len(m)>0 else 0)
        else:
            row.append("" if any(k in col.lower() for k in ["name","brand","cpu","gpu","type","company","os"]) else 0)

query_df = pd.DataFrame([row], columns=expected_cols)

# ------------------ RUN MODEL ------------------
<<<<<<< HEAD
predicted=None
if predict_clicked:
    pred = rf.predict(query_df)[0]
    if pred<=50: pred=np.exp(pred)
    predicted = pred/USD_RATE
=======
predicted = None
live_price = None
badge_html = "" # For the Value for Money score

if predict_clicked:
    pred = rf.predict(query_df)[0]
    if pred <= 50: pred = np.exp(pred)
    predicted = pred / USD_RATE
    
    # 1. Fetch live price
    print()
    search_query = f"{company} {type_name} {cpu} {ram}GB RAM"
    live_price = get_live_price(search_query)
    
    # 2. Logic for "Value for Money" Score
    if live_price:
        diff = predicted - live_price
        if diff > 50:
            badge_text, color = "🔥 GREAT DEAL", "#28a745"
        elif diff < -50:
            badge_text, color = "⚠️ OVERPRICED", "#dc3545"
        else:
            badge_text, color = "✅ FAIR PRICE", "#007bff"
            
        badge_html = f'<div style="background:{color}; color:white; padding:4px 12px; border-radius:20px; display:inline-block; font-size:12px; font-weight:bold; margin-bottom:10px;">{badge_text}</div>'
>>>>>>> 912b6f5 (Update logic to validate output against API data)

# -------------- INJECT CSS FOR SIDEBAR BUTTON (blue) --------------
# This styles the Streamlit sidebar button so it appears blue like your left UI button.
st.markdown(
    """
    <style>
    /* broadly target Streamlit button elements to give the sidebar button a blue style */
    .stButton>button {
        background-color: #2f6fe1 !important;
        color: #fff !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        border: none !important;
        box-shadow: none !important;
    }
    .stButton>button:hover { opacity: 0.95; }
    /* optional: make the sidebar area slightly separated visually */
    .css-1dp5vir { padding-top: 8px; } /* class may vary by Streamlit version; left as gentle override */
    </style>
    """,
    unsafe_allow_html=True,
)

<<<<<<< HEAD
# ------------------ HTML UI (NO CHATGPT IMAGES) ------------------
HTML_UI = f"""
<style>
body {{
    font-family: Inter, sans-serif;
}}
.container {{
    display:flex;
    gap:30px;
    margin-top:20px;
}}
.right-col {{
    flex:1;
    display:flex;
    flex-direction:column;
    gap:20px;
}}
.card {{
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0 8px 24px rgba(0,0,0,0.06);
}}
.price {{
    font-size:34px;
    font-weight:700;
    margin-top:10px;
}}
.range {{
    margin-top:6px;
    color:#555;
}}
</style>

<div class="container">

    <!-- ONLY RIGHT SIDE NOW -->
    <div class="right-col">

        <!-- SUMMARY CARD -->
        <div class="card">
            <h4>Laptop Summary Preview</h4>
            <div><strong>Brand:</strong> {company}</div>
            <div><strong>Type:</strong> {type_name}</div>
            <div><strong>OS:</strong> {os_name}</div>
            <div><strong>SSD:</strong> {ssd} GB</div>
            <div><strong>RAM:</strong> {ram} GB</div>
            <div><strong>Display:</strong> {screen_size} / {resolution}</div>
            <div><strong>CPU:</strong> {cpu}</div>
            <div><strong>GPU:</strong> {gpu}</div>
            <div><strong>Weight:</strong> {weight} kg</div>
            <div><strong>Touchscreen:</strong> {touch}</div>
            <div><strong>IPS:</strong> {ips}</div>
            <div><strong>HDD:</strong> {hdd} GB</div>
            
        </div>

        <!-- PRICE CARD -->
        <div class="card">
            <h4>Estimated Price</h4>
            <div class="price">{ "$"+format(predicted,",.2f") if predicted is not None else "—" }</div>
            <div class="range">
                { f"Range: ${predicted*0.9:,.2f} — ${predicted*1.1:,.2f}" if predicted else "" }
            </div>
        </div>

=======
# ------------------ HTML UI ------------------

HTML_UI = f"""
<style>
body {{ font-family: Inter, sans-serif; }}
.container {{ display:flex; gap:30px; margin-top:20px; }}
.right-col {{ flex:1; display:flex; flex-direction:column; gap:20px; }}
.card {{ background:white; padding:20px; border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,0.06); }}
.price {{ font-size:34px; font-weight:700; margin-top:10px; }}
.range {{ margin-top:6px; color:#555; }}
</style>

<div class="container">
    <div class="right-col">
        <div class="card">
            <h4>Laptop Summary Preview</h4>
            <div><strong>Brand:</strong> {company}</div>
            <div><strong>Specs:</strong> {cpu} | {ram}GB RAM | {ssd}GB SSD</div>
            <div><strong>Display:</strong> {screen_size}" {resolution}</div>
        </div>

        <div class="card">
            <h4>Estimated Price</h4>
            {badge_html}
            <div class="price">{ "$" + format(predicted, ",.2f") if predicted is not None else "—" }</div>
            
            <div class="range">
                <strong>Market Low:</strong> { f"${live_price:,.2f}" if live_price else "Searching market..." if predict_clicked else "—" }
            </div>
            <div class="range" style="font-size: 12px; color: #888;">
                { f"AI Confidence Range: ${predicted*0.9:,.2f} - ${predicted*1.1:,.2f}" if predicted else "" }
            </div>
        </div>
>>>>>>> 912b6f5 (Update logic to validate output against API data)
    </div>
</div>
"""

<<<<<<< HEAD
components.html(HTML_UI, height=700)

=======
components.html(HTML_UI, height=500)
>>>>>>> 912b6f5 (Update logic to validate output against API data)
