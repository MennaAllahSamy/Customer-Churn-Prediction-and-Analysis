import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# -------------------------------------------------------------------
# 1) Load scaler + 4 models from ml_model.pkl
# -------------------------------------------------------------------
@st.cache_resource
def load_all():
    # The tuple order must match the dump order:
    # (scaler, svm_model, decision_tree, rf, log_reg)
    scaler, svm_model, decision_tree, rf, log_reg = joblib.load("ML_MODELS.pkl")
    return scaler, svm_model, decision_tree, rf, log_reg

scaler, svm_model, decision_tree, rf, log_reg = load_all()

# -------------------------------------------------------------------
# 2) UI Header
# -------------------------------------------------------------------
st.title("💡 Customer Churn Prediction App")
st.markdown("Predict whether a customer will churn using one of four models.")
st.markdown("---")

# -------------------------------------------------------------------
# 3) Input Features
# -------------------------------------------------------------------
st.subheader("📊 Enter Customer Details")

subscription_age   = st.slider("Subscription Age (months)", 0, 72, 12)
bill_avg           = st.number_input("Average Bill", min_value=0.0, max_value=500.0, step=1.0)
remaining_contract = st.number_input("Remaining Contract (months)", min_value=0, max_value=72, step=1)
download_avg       = st.number_input("Average Download Speed (Mbps)", min_value=0.0, max_value=200.0, step=1.0)
upload_avg         = st.number_input("Average Upload Speed (Mbps)",   min_value=0.0, max_value=100.0, step=1.0)

# Scale the inputs
input_arr    = np.array([[subscription_age, bill_avg, remaining_contract, download_avg, upload_avg]])
scaled_input = scaler.transform(input_arr)

# -------------------------------------------------------------------
# 4) Prediction helper
# -------------------------------------------------------------------
def make_prediction(model, data):
    return "Churn" if model.predict(data)[0] == 1 else "No Churn"

# -------------------------------------------------------------------
# 5) Model Selection Buttons
# -------------------------------------------------------------------
st.markdown("### 🤖 Choose Model to Predict")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    if st.button("📈 SVM"):
        res = make_prediction(svm_model, scaled_input)
        st.success(f"SVM says: **{res}**")

with col2:
    if st.button("🌳 Decision Tree"):
        res = make_prediction(decision_tree, scaled_input)
        st.success(f"Decision Tree says: **{res}**")

with col3:
    if st.button("🌲 Random Forest"):
        res = make_prediction(rf, scaled_input)
        st.success(f"Random Forest says: **{res}**")

with col4:
    if st.button("🔍 Logistic Regression"):
        res = make_prediction(log_reg, scaled_input)
        st.success(f"Logistic Regression says: **{res}**")

# -------------------------------------------------------------------
# 6) Footer
# -------------------------------------------------------------------
st.markdown("---")
st.markdown("Customer Churn Prediction App")
