import streamlit as st
st.set_page_config(layout="wide")
from pathlib import Path
import pickle
import json
import numpy as np
import math

MODEL_FILE = Path("banglore_home_prices_model.pickle")
COLUMNS_FILE = Path("columns.json")

st.set_page_config(page_title="🏡 Bangalore House Price", page_icon="🏡", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #66c6f7 0%, #8ed7fb 45%, #cfeefd 100%);
        background-attachment: fixed;
        min-height: 100vh;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        text-align: center;
        color: #052f5f;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    

    .input-card .stMarkdown p, .input-card .stTextInput, .input-card .stSelectbox {
        margin-bottom: 6px !important;
    }

    .input-card .stTextInput>div>div>input,
    .input-card .stNumberInput>div>input,
    .input-card .stSelectbox>div>div>select {
        height:44px;
        font-size:16px;
        border-radius:8px;
        padding-left:12px;
    }

    .input-card .stSelectbox, .input-card .stTextInput, .input-card .stNumberInput {
        width: 100%;
    }

    .estimate-button .stButton>button {
        background: linear-gradient(180deg,#8ee07a,#6ccf4a);
        color: #06400a;
        font-weight: 700;
        height:40px;
        width:140px;
        border-radius:8px;
        border: none;
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    .estimate-button .stButton>button:hover {
        filter:brightness(0.98);
    }

    .result-container {
        display:flex;
        justify-content:center;
        width:100%;
        margin-top: 18px;
    }
    .result-box {
        background: linear-gradient(180deg,#ffd85a 0%, #f7cc42 100%);
        padding: 16px 28px;
        border-radius: 14px;
        font-size: 22px;
        font-weight: 800;
        text-align: center;
        color: #111;
        box-shadow: 0 14px 30px rgba(0,0,0,0.12);
        min-width: 260px;
        display:inline-block;
    }

    @media (max-width: 720px) {
        .input-card { padding: 16px; width: 92%; }
        .result-box { min-width: 220px; font-size: 20px; padding: 14px 20px; }
    }

    .block-container label {
        margin-bottom: -20px !important;
    }

    .stSelectbox, .stNumberInput, .stRadio {
        margin-top: -10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Bangalore House Price Predictor</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        st.error(f"`{MODEL_FILE.name}` not found. Place your trained model file in this directory.")
        st.stop()
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_columns():
    if not COLUMNS_FILE.exists():
        return None
    with open(COLUMNS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    cols = data.get("data_columns") or data.get("columns") or data.get("columns_list")
    if not isinstance(cols, list):
        return None
    return cols

model = load_model()
cols = load_columns()
location_cols = cols[3:] if cols and len(cols) >= 4 else []

st.markdown('<div class="input-card">', unsafe_allow_html=True)

st.markdown("**Area (Square Feet)**")
area_raw = st.text_input("", placeholder="Enter area in sqft (e.g. 1000)", key="area_input")
try:
    sqft = float(area_raw) if area_raw and area_raw.strip() != "" else None
except:
    sqft = None

st.markdown("**BHK**")
bhk_options = ["— select —", 1, 2, 3, 4, 5]
bhk_choice_raw = st.selectbox("", options=bhk_options, index=0, key="bhk_select")
bhk_choice = None if bhk_choice_raw == "— select —" else int(bhk_choice_raw)

st.markdown("**Bath**")
bath_options = ["— select —", 1, 2, 3, 4, 5]
bath_choice_raw = st.selectbox("", options=bath_options, index=0, key="bath_select")
bath_choice = None if bath_choice_raw == "— select —" else int(bath_choice_raw)

st.markdown("**Location**")
if location_cols:
    location = st.selectbox("", options=["— select location —"] + location_cols, index=0)
    if location == "— select location —":
        location = ""
else:
    location = st.text_input("", placeholder="Ex: Akshaya Nagar")

st.markdown('<div style="height:3px"></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 0.6, 1])
with col2:
    st.markdown('<div class="estimate-button">', unsafe_allow_html=True)
    predict_clicked = st.button("Estimate Price", key="estimate_btn")
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

def build_feature_vector(location_str: str, sqft_val: float, bath_val: int, bhk_val: int, columns_list):
    if columns_list is None:
        raise ValueError("columns.json missing or invalid. See instructions below to create it.")
    cols_local = columns_list
    x = np.zeros(len(cols_local), dtype=float)
    x[0] = float(sqft_val)
    x[1] = float(bath_val)
    x[2] = float(bhk_val)
    matches = [i for i, c in enumerate(cols_local) if c == location_str]
    if len(matches) == 0:
        return x, -1
    loc_index = matches[0]
    x[loc_index] = 1.0
    return x, loc_index

def format_inr(amount: float) -> str:
    try:
        integer_part = int(math.floor(abs(amount)))
        decimals = int(round((abs(amount) - integer_part) * 100))
        s = str(integer_part)
        if len(s) <= 3:
            int_fmt = s
        else:
            last3 = s[-3:]
            rest = s[:-3]
            parts = []
            while len(rest) > 2:
                parts.append(rest[-2:])
                rest = rest[:-2]
            if rest:
                parts.append(rest)
            parts.reverse()
            int_fmt = ",".join(parts) + "," + last3
        sign = "-" if amount < 0 else ""
        return f"{sign}₹ {int_fmt}.{decimals:02d}"
    except Exception:
        return f"₹ {amount:,.2f}"

if predict_clicked:
    if not location or str(location).strip() == "":
        st.error("Please select a location.")
    elif sqft is None:
        st.error("Please enter a valid area (sqft).")
    elif bhk_choice is None:
        st.error("Please select BHK.")
    elif bath_choice is None:
        st.error("Please select number of bathrooms.")
    else:
        try:
            X_vec, loc_idx = build_feature_vector(location, sqft, bath_choice, bhk_choice, cols)
        except Exception as e:
            st.error(f"Failed to prepare features: {e}")
        else:
            with st.spinner("Estimating price..."):
                try:
                    pred = model.predict([X_vec])[0]
                    pred_value = float(pred)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    pred_value = None

            if pred_value is not None:
                formatted = format_inr(pred_value)
                st.markdown(
                    f"""
                    <div class="result-container">
                        <div class="result-box">{formatted} Lakh</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if loc_idx == -1:
                    st.warning("Selected location not found in training data — result may be unreliable.")
