import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Page Config (MUST BE FIRST) ---
st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️", layout="wide")

# --- Configuration ---
MODEL_FILE = 'fraud_model.pkl'

# --- 2. Load Model ---
@st.cache_resource
def load_model_artifacts():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)

# Load the model artifacts
artifacts = load_model_artifacts()

# --- Header ---
st.title("🛡️ FraudGuard AI Detection System")
st.markdown("""
**Interactive Dashboard:** Adjust the sliders below to simulate different transaction patterns.
The model will analyze the **Amount** and **Technical Features (V1-V28)** to predict fraud probability.
""")
st.markdown("---")

# --- Check if model exists ---
if artifacts is None:
    st.error("❌ Model file not found!")
    st.warning("Please run 'train_model.py' first to generate the model.")
    st.stop()

model = artifacts['model']
scaler = artifacts['scaler']
feature_names = artifacts['feature_names']

# --- Sidebar Inputs ---
st.sidebar.header("1. Transaction Details")

# --- DOLLAR AMOUNT INPUT ---
amount = st.sidebar.number_input(
    "Transaction Amount ($)", 
    min_value=0.0, 
    value=150.0, 
    step=10.0,
    help="The dollar value of the transaction."
)

st.sidebar.header("2. Technical Features (V-Input)")
st.sidebar.info("V1-V28 are anonymized Principal Components (PCA). Tweaking these simulates different hidden transaction behaviors (e.g., location, IP address changes).")

# Critical features sliders
v14 = st.sidebar.slider("V14 (High Impact)", -5.0, 5.0, 0.0)
v4 = st.sidebar.slider("V4 (High Impact)", -5.0, 5.0, 0.0)
v11 = st.sidebar.slider("V11 (High Impact)", -5.0, 5.0, 0.0)

# --- Logic: Generate Full Input Vector ---
input_data = {}

# 1. Fill background noise (REALISTIC DATA)
# Instead of 0, we use small random numbers to simulate natural variance
# This makes the Data Inspector look like a real transaction
for col in feature_names:
    if col.startswith('V'):
        input_data[col] = np.random.normal(0, 0.5) 

# 2. Overwrite with user slider values
input_data['V14'] = v14
input_data['V4'] = v4
input_data['V11'] = v11

# 3. Scale the Amount
scaled_amount = scaler.transform([[amount]])[0][0]
input_data['Amount_Scaled'] = scaled_amount

# 4. Create the Data Frame for the model
input_df = pd.DataFrame([input_data], columns=feature_names)

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Transaction Analysis")
    
    # The "Analyze" Button
    if st.button("Run Fraud Detection Model", type="primary", use_container_width=True):
        
        # Get Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] # Probability of Class 1 (Fraud)

        st.divider()
        
        # Result Logic
        if prediction == 1:
            st.error("🚨 **FRAUD DETECTED**")
            st.markdown(f"**Confidence:** The model is **{probability:.2%}** sure this is fraud.")
            st.markdown("### ⚠️ Recommendation: Block Transaction")
        else:
            st.success("✅ **LEGITIMATE TRANSACTION**")
            st.markdown(f"**Confidence:** The model assesses this as safe. (Fraud Risk: {probability:.2%})")
            st.markdown("### 👍 Recommendation: Approve")
            
        # Visual Progress Bar
        st.write("Fraud Probability Score:")
        st.progress(int(probability * 100))

with col2:
    st.subheader("🔢 Data Inspector")
    st.caption("This is the exact mathematical vector sent to the AI model (includes random noise):")
    
    # Transpose the dataframe to make it look like a vertical list
    st.dataframe(input_df.T.style.format("{:.4f}"), height=400)

st.markdown("---")