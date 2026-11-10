
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import skfuzzy as fuzz

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AirSense | AQI Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.main {
    background-color: #f5f8fc;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e1e8f0;
}

.stButton>button {
    width: 100%;
    background-color: #087f8c;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}

h1 {
    color: #087f8c;
}

h2, h3 {
    color: #17324d;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD TRAINED ANN MODEL
# ==========================================

MODEL_PATH = "models/air_quality_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)

saved = load_model()

# ==========================================
# 4. AQI CATEGORY FUNCTION
# ==========================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good", "🟢"

    elif aqi <= 100:
        return "Satisfactory", "🟡"

    elif aqi <= 200:
        return "Moderate", "🟠"

    elif aqi <= 300:
        return "Poor", "🔴"

    elif aqi <= 400:
        return "Very Poor", "🟣"

    else:
        return "Severe", "⚫"


# ==========================================
# 5. HEALTH RECOMMENDATIONS
# ==========================================

def get_health_advice(category):

    advice = {
        "Good":
            "Air quality is good. Normal outdoor activities are suitable.",

        "Satisfactory":
            "Generally acceptable air quality. Sensitive individuals should monitor symptoms.",

        "Moderate":
            "People with respiratory or heart conditions should reduce prolonged outdoor exertion.",

        "Poor":
            "Reduce prolonged outdoor exposure, especially for sensitive groups.",

        "Very Poor":
            "Avoid prolonged outdoor exertion and follow local health advisories.",

        "Severe":
            "Avoid outdoor exertion and follow official emergency air-quality guidance."
    }

    return advice[category]


# ==========================================
# 6. FUZZY LOGIC CLASSIFICATION
# ==========================================

def fuzzy_prediction(aqi):

    # Input universe
    x = np.arange(0, 501, 1)

    # Output universe: pollution risk score
    y = np.arange(0, 101, 1)

    # AQI membership functions
    good = fuzz.trapmf(x, [0, 0, 25, 50])
    satisfactory = fuzz.trimf(x, [25, 75, 100])
    moderate = fuzz.trimf(x, [75, 150, 200])
    poor = fuzz.trimf(x, [150, 250, 300])
    very_poor = fuzz.trimf(x, [250, 350, 400])
    severe = fuzz.trapmf(x, [350, 425, 500, 500])

    # Risk output membership functions
    low = fuzz.trapmf(y, [0, 0, 10, 25])
    mild = fuzz.trimf(y, [15, 30, 45])
    medium = fuzz.trimf(y, [35, 50, 65])
    high = fuzz.trimf(y, [55, 70, 85])
    very_high = fuzz.trimf(y, [75, 88, 96])
    extreme = fuzz.trapmf(y, [90, 96, 100, 100])

    # Fuzzy rules
    rules = [
        (fuzz.interp_membership(x, good, aqi), low),
        (fuzz.interp_membership(x, satisfactory, aqi), mild),
        (fuzz.interp_membership(x, moderate, aqi), medium),
        (fuzz.interp_membership(x, poor, aqi), high),
        (fuzz.interp_membership(x, very_poor, aqi), very_high),
        (fuzz.interp_membership(x, severe, aqi), extreme)
    ]

    # Aggregate fuzzy outputs
    aggregated = np.zeros_like(y, dtype=float)

    for strength, membership in rules:
        aggregated = np.maximum(
            aggregated,
            np.minimum(strength, membership)
        )

    if np.max(aggregated) == 0:
        return 0

    # Defuzzification
    risk_score = fuzz.defuzz(
        y,
        aggregated,
        "centroid"
    )

    return round(float(risk_score), 2)


# ==========================================
# 7. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🌍 AirSense")

    st.caption("AI-Based Air Quality Monitoring")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "AQI Prediction",
            "Prediction History",
            "About Project"
        ]
    )

    st.divider()

    st.info(
        "Powered by Artificial Neural Networks "
        "and Fuzzy Logic."
    )

# ==========================================
# 8. HEADER
# ==========================================

st.title("🌍 AirSense")
st.subheader("Intelligent Air Quality Prediction System")

st.markdown(
    "Predict air pollution levels using "
    "Artificial Intelligence and Soft Computing."
)

st.divider()

# ==========================================
# 9. CHECK MODEL
# ==========================================

if saved is None:

    st.error(
        "Trained ANN model not found!"
    )

    st.warning(
        "Please run: python train_model.py"
    )

    st.stop()

model = saved["model"]
features = saved["features"]

if not hasattr(model, "predict"):
    st.error("Invalid model file.")
    st.stop()

# ==========================================
# 10. SESSION HISTORY
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# 11. DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.header("Air Quality Dashboard")

    st.write(
        "Welcome to AirSense. "
        "Use the AQI Prediction page to estimate "
        "air quality from pollutant concentrations."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "AI Model",
        "ANN",
        "Trained model"
    )

    col2.metric(
        "Soft Computing",
        "Fuzzy Logic",
        "Enabled"
    )

    col3.metric(
        "Features",
        len(features),
        "Pollutant parameters"
    )

    st.divider()

    st.subheader("Supported Pollution Parameters")

    parameter_info = pd.DataFrame({
        "Parameter": features,
        "Description": [
            "Fine particulate matter",
            "Coarse particulate matter",
            "Nitric oxide",
            "Nitrogen dioxide",
            "Nitrogen oxides",
            "Ammonia",
            "Carbon monoxide",
            "Sulphur dioxide",
            "Ozone"
        ]
    })

    st.dataframe(
        parameter_info,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Navigate to AQI Prediction to enter "
        "pollutant values and generate predictions."
    )

# ==========================================
# 12. AQI PREDICTION PAGE
# ==========================================

elif page == "AQI Prediction":

    st.header("Air Quality Prediction")

    st.write(
        "Enter the pollutant concentrations "
        "to predict the Air Quality Index."
    )

    st.caption(
        "Use the same units as your Kaggle city_day.csv dataset. "
        "For the standard dataset, CO is in mg/m³ and "
        "the other listed pollutants are in µg/m³."
    )

    st.divider()

    st.subheader("Enter Pollution Parameters")

    default_values = {
        "PM2.5": 35.0,
        "PM10": 60.0,
        "NO": 10.0,
        "NO2": 20.0,
        "NOx": 30.0,
        "NH3": 10.0,
        "CO": 1.0,
        "SO2": 10.0,
        "O3": 30.0
    }

    input_values = {}

    col1, col2, col3 = st.columns(3)

    columns = [col1, col2, col3]

    for i, feature in enumerate(features):

        with columns[i % 3]:

            input_values[feature] = st.number_input(
                feature,
                min_value=0.0,
                max_value=10000.0,
                value=float(default_values.get(feature, 10.0)),
                step=1.0,
                key="input_" + feature
            )

    st.divider()

    if st.button(
        "🔍 Predict Air Quality",
        type="primary"
    ):

        try:

            input_df = pd.DataFrame(
                [input_values],
                columns=features
            )

            # ANN prediction
            raw_prediction = float(
                model.predict(input_df)[0]
            )

            # AQI display range
            aqi = int(
                np.clip(round(raw_prediction), 0, 500)
            )

            category, emoji = get_aqi_category(aqi)

            # Fuzzy logic
            fuzzy_risk = fuzzy_prediction(aqi)

            # Save history
            st.session_state.history.append({
                "Date": pd.Timestamp.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "AQI": aqi,
                "Category": category
            })

            st.success(
                "AQI prediction generated successfully!"
            )

            st.divider()

            st.header("Prediction Results")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Predicted AQI",
                aqi
            )

            c2.metric(
                "Air Quality",
                category
            )

            c3.metric(
                "Fuzzy Risk Score",
                f"{fuzzy_risk}/100"
            )

            st.markdown(
                f"## {emoji} {category} Air Quality"
            )

            if raw_prediction < 0 or raw_prediction > 500:
                st.warning(
                    "The ANN prediction was outside the "
                    "displayed AQI range. It has been clipped "
                    "to 0–500; interpret this result cautiously."
                )

            # AQI gauge-style progress
            st.subheader("AQI Level")

            st.progress(aqi / 500)

            st.caption(
                f"Predicted AQI: {aqi} out of 500"
            )

            st.divider()

            # Health recommendation
            st.subheader("Health Recommendations")

            st.info(
                get_health_advice(category)
            )

            st.divider()

            # Input visualization
            st.subheader("Pollutant Concentrations")

            chart_df = pd.DataFrame({
                "Pollutant": list(input_values.keys()),
                "Concentration": list(input_values.values())
            })

            st.bar_chart(
                chart_df.set_index("Pollutant")
            )

            # Results table
            st.subheader("Input Data Summary")

            st.dataframe(
                chart_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {str(e)}"
            )

# ==========================================
# 13. PREDICTION HISTORY
# ==========================================

elif page == "Prediction History":

    st.header("Prediction History")

    if len(st.session_state.history) == 0:

        st.info(
            "No predictions available yet. "
            "Make a prediction first."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("AQI History Chart")

        st.line_chart(
            history_df.set_index("Date")["AQI"]
        )

        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Prediction History",
            data=csv,
            file_name="airsense_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()

# ==========================================
# 14. ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.header("About AirSense")

    st.write("""
    AirSense is an AI-based air quality prediction
    system developed using Soft Computing techniques.

    The project combines Artificial Neural Networks
    and Fuzzy Logic to estimate and interpret
    air pollution levels.
    """)

    st.subheader("Technologies Used")

    technologies = [
        "Python",
        "Streamlit",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Artificial Neural Networks",
        "Fuzzy Logic",
        "Matplotlib"
    ]

    for technology in technologies:
        st.write("✅", technology)

    st.subheader("Project Objectives")

    st.write("""
    1. Predict AQI from pollutant concentrations.
    2. Apply ANN for nonlinear prediction.
    3. Apply Fuzzy Logic for pollution-risk assessment.
    4. Visualize prediction results.
    5. Provide general health recommendations.
    """)

    st.warning(
        "This is an educational prediction system. "
        "It is not an official AQI monitoring service. "
        "Actual CPCB AQI calculation uses pollutant-specific "
        "sub-indices and the maximum applicable sub-index."
    )

# ==========================================
# 15. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:gray;">
        🌍 AirSense | AI-Based Air Quality Prediction System
        <br>
        Powered by Artificial Intelligence and Soft Computing
    </div>
    """,
    unsafe_allow_html=True
)
