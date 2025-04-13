import streamlit as st
from agrofit import AgroFit
from yield_predictor import YieldPredictor
import warnings

warnings.filterwarnings("ignore", message=".libiomp.*libomp.")

# Load Models
data_path = "enlarged_agriculture_dataset.csv"
agrofit = AgroFit(data_path=data_path, model_path="agrofit_model.pkl")
yield_predictor = YieldPredictor(data_path=data_path, model_path="yield_predictor_model.pkl")

# Page Config
st.set_page_config(page_title="Farm Assist AI", layout="wide")

# Background and Styling
background_image_url = "https://images.pexels.com/photos/338936/pexels-photo-338936.jpeg"

st.markdown(f"""
    <style>
    .stApp {{
        background: url("{background_image_url}") no-repeat center center fixed;
        background-size: cover;
        font-family: 'Segoe UI', sans-serif;
    }}

    /* Top Navigation */
    .topbar {{
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1rem 0 2rem 0;
        background-color: transparent !important;
        box-shadow: none !important;
    }}

    /* Headers */
    h1 {{
        color: #ffffff !important;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }}
    h2.subhead {{
        color: #97ff74;
        text-align: center;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 400;
        text-color:#ffffff !important;
    }}

    h3, label {{
        color: #e0ffe0 !important;
    }}

    /* Text input field */
    .stTextInput > div > input {{
        background-color: #1e1e1e !important;
        border-radius: 15px !important;
        padding: 8px !important;
        box-shadow: 0 0 6px rgba(0, 255, 128, 0.3);    
        border: 1px solid #ffffff !important; 
        color: #ffffff !important;
    }}

    /* Selectbox - selected value inside box */
    div[data-baseweb="select"] > div {{
        background-color: #1e1e1e !important;
        border-radius: 15px !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
    }}

    /* Selectbox - dropdown options */
    div[data-baseweb="popover"] li {{
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }}

    /* Slider and its text */
    .stSlider {{
        background-color: #1e1e1e !important;
        border-radius: 15px !important;
        padding: 8px !important;
        box-shadow: 0 0 6px rgba(0, 255, 128, 0.3);    
        border: 1px solid #ffffff !important; 
    }}
    .stSlider > div > div > span {{
        color: #ffffff !important;
    }}

    /* Buttons */
    .stButton > button {{
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        color: white;
        background-color: #1e3d59;
        transition: 0.3s;
    }}
    .stButton > button:hover {{
        background-color: #2e8b57 !important;
    }}
    .stButton > button.active {{
        background-color: #4CAF50 !important;
        border: 2px solid #ffffff;
    }}
    </style>
""", unsafe_allow_html=True)


# Title & Subtitle
st.markdown("<h1>🌾 Farm Assist AI</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subhead'>Smart  Agriculture  Assistance</h2> ", unsafe_allow_html=True)

# Navigation
st.markdown("<div class='topbar'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    agrofit_btn = st.button("🌿 Crop Recomendation", key="subtype")
with col2:
    cond_btn = st.button("🌱 Recommended Conditions", key="condition")
with col3:
    yield_btn = st.button("📈 Yield Prediction", key="yield")
st.markdown("</div>", unsafe_allow_html=True)

# Page routing
if 'page' not in st.session_state:
    st.session_state.page = 'agrofit'
if agrofit_btn:
    st.session_state.page = 'agrofit'
elif cond_btn:
    st.session_state.page = 'conditions'
elif yield_btn:
    st.session_state.page = 'yield'

# Main Content Area
st.markdown("<div class='overlay'>", unsafe_allow_html=True)

# AgroFit Subtype
if st.session_state.page == 'agrofit':
    st.subheader("🧬 Crop Subtype Recommendation")
    N = st.slider("Nitrogen (ppm)", 10, 200, 40)
    P = st.slider("Phosphorus (ppm)", 10, 200, 30)
    K = st.slider("Potassium (ppm)", 10, 200, 50)
    temp = st.slider("Temperature (°C)", 5, 50, 25)
    humidity = st.slider("Humidity (%)", 10, 100, 60)
    pH = st.slider("pH Level", 2.0, 14.0, 6.5)
    rainfall = st.slider("Rainfall (mm)", 50, 1000, 300)

    if st.button("🌱 Get Subtype Recommendation", key="get_subtype"):
        result = agrofit.recommend_subtype([N, P, K, temp, humidity, pH, rainfall])
        st.success(f"✅ Recommended Subtype: *{result}*")

# Recommended Conditions
elif st.session_state.page == 'conditions':
    st.subheader("🌿 Recommended Conditions")
    subtype = st.text_input("Enter Crop Subtype")
    variety = st.text_input("Enter Variety")

    if st.button("🔍 Get Recommended Conditions", key="get_conditions"):
        result = agrofit.recommend_conditions(subtype, variety)
        if isinstance(result, dict):
            st.write("📋 Recommended Environmental Ranges:")
            for k, v in result.items():
                st.write(f"✅ *{k}*: {v}")
        else:
            st.error(result)

# Yield Prediction
elif st.session_state.page == 'yield':
    st.subheader("📈 Yield Prediction")
    options = yield_predictor.get_categorical_options()
    user_input = {}

    for cat_feat in options:
        user_input[cat_feat] = st.selectbox(f"{cat_feat}", options[cat_feat], key=f"cat_{cat_feat}")

    numeric_features = {
        'Nitrogen Content (ppm)': (10, 300, 40),
        'Phosphorus Content (ppm)': (20, 200, 20),
        'Potassium Content (ppm)': (10, 200, 30),
        'Temperature (°C)': (5, 50, 25),
        'Humidity (%)': (10, 100, 60),
        'Rainfall (mm)': (50, 1000, 300),
        'pH Level': (2.0, 14.0, 6.5),
        'Altitude (m)': (200, 3000, 200),
        'Sunlight Hours (per day)': (2, 24, 6),
        'Growing Period (days)': (10, 300, 90),
        'Area (ha)': (0, 500, 1)
    }

    for feature, (min_val, max_val, default) in numeric_features.items():
        user_input[feature] = st.slider(feature, min_val, max_val, default)

    if st.button("📊 Predict Yield", key="predict"):
        predicted_yield = yield_predictor.predict_yield(user_input)
        if predicted_yield:
            st.success(f"🌾 Predicted Yield: *{predicted_yield:.2f} kg/ha*")
        else:
            st.error("❌ Prediction failed. Please check your inputs.")

st.markdown("</div>", unsafe_allow_html=True)
