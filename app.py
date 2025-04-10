import streamlit as st
from agrofit import AgroFit
from yield_predictor import YieldPredictor
import warnings

warnings.filterwarnings("ignore", message=".*libiomp.*libomp.*")


# Load models
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
    }}
    .top-container {{
        background-color: rgba(0, 100, 0, 0.85);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }}
    .model-button {{
        margin: 0.5rem;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        cursor: pointer;
        color: white;
        background-color: #228B22;
    }}
    .model-button:hover {{
        background-color: #2e8b57;
    }}
    .main-container {{
        background-color: rgba(0, 0, 0, 0.65);
        padding: 2rem;
        border-radius: 15px;
    }}
    h2, label {{
        color: #e0ffe0 !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #e0ffe0;'>🌾 Farm Assist AI</h1>", unsafe_allow_html=True)

# Top Container - Selection Buttons
with st.container():
    st.markdown("<div class='top-container'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        agrofit_btn = st.button("🧬 AgroFit Subtype", key="subtype_btn")
    with col2:
        cond_btn = st.button("🌿 Recommended Conditions", key="condition_btn")
    with col3:
        yield_btn = st.button("📈 Yield Prediction", key="yield_btn")
    st.markdown("</div>", unsafe_allow_html=True)

# State Tracking
if 'page' not in st.session_state:
    st.session_state.page = 'agrofit'

if agrofit_btn:
    st.session_state.page = 'agrofit'
elif cond_btn:
    st.session_state.page = 'conditions'
elif yield_btn:
    st.session_state.page = 'yield'

# Main Display Section
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    if st.session_state.page == 'agrofit':
        st.subheader("🧬 Crop Subtype Recommendation")
        N = st.slider("Nitrogen (ppm)", 0, 200, 40, key="N")
        P = st.slider("Phosphorus (ppm)", 0, 200, 30, key="P")
        K = st.slider("Potassium (ppm)", 0, 200, 50, key="K")
        temp = st.slider("Temperature (°C)", 0, 50, 25, key="temp")
        humidity = st.slider("Humidity (%)", 0, 100, 60, key="humidity")
        pH = st.slider("pH Level", 0.0, 14.0, 6.5, key="ph")
        rainfall = st.slider("Rainfall (mm)", 0, 1000, 300, key="rainfall")

        if st.button("🌱 Get Subtype Recommendation", key="rec_subtype"):
            result = agrofit.recommend_subtype([N, P, K, temp, humidity, pH, rainfall])
            st.success(f"✅ Recommended Subtype: **{result}**")

    elif st.session_state.page == 'conditions':
        st.subheader("🌿 Recommended Conditions")
        subtype = st.text_input("Enter Crop Subtype", key="subtype_input")
        variety = st.text_input("Enter Variety", key="variety_input")

        if st.button("🔍 Get Recommended Conditions", key="rec_cond"):
            result = agrofit.recommend_conditions(subtype, variety)
            if isinstance(result, dict):
                st.write("📋 Recommended Environmental Ranges:")
                for k, v in result.items():
                    st.write(f"✅ **{k}**: {v}")
            else:
                st.error(result)

    elif st.session_state.page == 'yield':
        st.subheader("📈 Yield Prediction")
        options = yield_predictor.get_categorical_options()
        user_input = {}

        # Dropdowns
        for cat_feat in options:
            user_input[cat_feat] = st.selectbox(f"{cat_feat}", options[cat_feat], key=f"cat_{cat_feat}")

        # Sliders
        numeric_features = {
            'Nitrogen Content (ppm)': (0, 300, 40),
            'Phosphorus Content (ppm)': (0, 200, 20),
            'Potassium Content (ppm)': (0, 200, 30),
            'Temperature (°C)': (0, 50, 25),
            'Humidity (%)': (0, 100, 60),
            'Rainfall (mm)': (0, 1000, 300),
            'pH Level': (0.0, 14.0, 6.5),
            'Altitude (m)': (0, 3000, 200),
            'Sunlight Hours (per day)': (0, 24, 6),
            'Growing Period (days)': (0, 300, 90),
            'Area (ha)': (0, 500, 1)
        }

        for feature, (min_val, max_val, default) in numeric_features.items():
            user_input[feature] = st.slider(feature, min_val, max_val, default, key=feature)

        if st.button("📊 Predict Yield", key="predict_yield"):
            predicted_yield = yield_predictor.predict_yield(user_input)
            if predicted_yield:
                st.success(f"🌾 Predicted Yield: **{predicted_yield:.2f} kg/ha**")
            else:
                st.error("❌ Prediction failed. Please check your inputs.")

    st.markdown("</div>", unsafe_allow_html=True)
