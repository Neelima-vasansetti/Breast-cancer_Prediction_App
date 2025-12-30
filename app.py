# app.py

import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import warnings
import time

warnings.filterwarnings("ignore")

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------------
# GLOBAL STYLES + ANIMATIONS
# ---------------------------------
st.markdown("""
<style>

/* Animated gradient background */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: #ffffff;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Title animation */
h1 {
    text-align: center;
    animation: slideFade 1.2s ease-in-out;
}

@keyframes slideFade {
    from {opacity: 0; transform: translateY(-30px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Input card */
.input-card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 10px 40px rgba(0,0,0,0.3);
    animation: fadeUp 1s ease;
}

@keyframes fadeUp {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 14px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.06);
    box-shadow: 0px 12px 30px rgba(255,75,75,0.6);
}

/* Result card */
.result-card {
    background: rgba(0,0,0,0.35);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    margin-top: 25px;
    animation: zoomIn 0.6s ease;
}

@keyframes zoomIn {
    from {transform: scale(0.8); opacity: 0;}
    to {transform: scale(1); opacity: 1;}
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Title
# ---------------------------------
st.title("🩺 Breast Cancer Prediction")
st.write(
    "This AI system analyzes tumor features and predicts whether the tumor is **Benign** or **Malignant**."
)

# ---------------------------------
# Load Model & Scaler
# ---------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("BC_model.h5")
    with open("BC_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ---------------------------------
# INPUT FEATURES
# ---------------------------------
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.subheader("🔬 Tumor Feature Inputs")

input_data = {
    'radius_mean': st.number_input('radius_mean', value=14.0),
    'texture_mean': st.number_input('texture_mean', value=20.0),
    'perimeter_mean': st.number_input('perimeter_mean', value=90.0),
    'area_mean': st.number_input('area_mean', value=600.0),
    'smoothness_mean': st.number_input('smoothness_mean', value=0.1),
    'compactness_mean': st.number_input('compactness_mean', value=0.15),
    'concavity_mean': st.number_input('concavity_mean', value=0.2),
    'concave points_mean': st.number_input('concave points_mean', value=0.1),
    'symmetry_mean': st.number_input('symmetry_mean', value=0.2),
    'fractal_dimension_mean': st.number_input('fractal_dimension_mean', value=0.06),

    'radius_se': st.number_input('radius_se', value=0.2),
    'texture_se': st.number_input('texture_se', value=1.0),
    'perimeter_se': st.number_input('perimeter_se', value=1.5),
    'area_se': st.number_input('area_se', value=20.0),
    'smoothness_se': st.number_input('smoothness_se', value=0.005),
    'compactness_se': st.number_input('compactness_se', value=0.02),
    'concavity_se': st.number_input('concavity_se', value=0.03),
    'concave points_se': st.number_input('concave points_se', value=0.01),
    'symmetry_se': st.number_input('symmetry_se', value=0.03),
    'fractal_dimension_se': st.number_input('fractal_dimension_se', value=0.004),

    'radius_worst': st.number_input('radius_worst', value=16.0),
    'texture_worst': st.number_input('texture_worst', value=25.0),
    'perimeter_worst': st.number_input('perimeter_worst', value=105.0),
    'area_worst': st.number_input('area_worst', value=800.0),
    'smoothness_worst': st.number_input('smoothness_worst', value=0.12),
    'compactness_worst': st.number_input('compactness_worst', value=0.2),
    'concavity_worst': st.number_input('concavity_worst', value=0.3),
    'concave points_worst': st.number_input('concave points_worst', value=0.15),
    'symmetry_worst': st.number_input('symmetry_worst', value=0.25),
    'fractal_dimension_worst': st.number_input('fractal_dimension_worst', value=0.08),
}

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# PREDICTION
# ---------------------------------
if st.button("🔍 Predict Cancer Type"):
    with st.spinner("🧠 AI is analyzing the tumor features..."):
        time.sleep(1)

        input_df = pd.DataFrame([input_data])

        # Feature safety check
        if list(input_df.columns) != list(scaler.feature_names_in_):
            st.error("❌ Feature mismatch between input and trained model.")
            st.stop()

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0][0]

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        predicted_class = "Malignant" if prediction > 0.5 else "Benign"
        color = "#ff4b2b" if predicted_class == "Malignant" else "#00ffae"

        st.markdown(f"""
        <div class="result-card">
            <h2>🧬 Prediction Result</h2>
            <h1 style="color:{color};">{predicted_class}</h1>
            <p style="font-size:18px;">
                Confidence Score: <b>{prediction:.4f}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("""
<hr style="opacity:0.2">
<p style="text-align:center; font-size:14px;">
Built with ❤️ using Streamlit & Deep Learning
</p>
""", unsafe_allow_html=True)
