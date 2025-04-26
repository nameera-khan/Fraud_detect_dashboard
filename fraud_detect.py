# app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Load model and scaler
model = pickle.load(open('fraud_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# App Title
st.title("🛡️ Fraudulent User Detection Dashboard")

# Sidebar
st.sidebar.header('Upload User Data')
uploaded_file = st.sidebar.file_uploader("Upload your user features CSV", type=["csv"])

# If file uploaded, read it
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.sidebar.write("Or using simulated user data.")
    # Create dummy data for demo
    data = pd.DataFrame({
        'Product_Price': np.random.randint(20, 400, 100),
        'Order_Quantity': np.random.randint(1, 3, 100),
        'Discount_Applied': np.random.uniform(0, 0.5, 100),
        'Days_to_Return_Corrected': np.random.randint(0, 90, 100),
        'Late_Return_Flag': np.random.randint(0, 2, 100),
        'Suspicious_Reason_Flag': np.random.randint(0, 2, 100),
        'Fast_Return_Flag': np.random.randint(0, 2, 100),
        'Suspicious_Score': np.random.randint(0, 4, 100),
        'User_Total_Orders': np.random.randint(1, 40, 100),
        'User_Total_Returns': np.random.randint(0, 30, 100),
        'User_Return_Rate': np.random.uniform(0, 1, 100),
        'User_Avg_Order_Value': np.random.randint(20, 400, 100),
        'User_Total_Spent': np.random.randint(500, 10000, 100),
        'High_Returner_Flag': np.random.randint(0, 2, 100)
    })

# Scale data
data_scaled = scaler.transform(data)

# Make predictions
pred_probs = model.predict_proba(data_scaled)[:, 1]
data['Fraud_Probability'] = pred_probs
data['Fraud_Label'] = np.where(data['Fraud_Probability'] > 0.5, 'Fraudulent', 'Normal')

# KPIs
total_users = len(data)
fraudulent_users = sum(data['Fraud_Label'] == 'Fraudulent')
fraud_rate = fraudulent_users / total_users * 100

# Layout KPIs
col1, col2, col3 = st.columns(3)

col1.metric("Total Users Reviewed", total_users)
col2.metric("Fraudulent Users Detected", fraudulent_users)
col3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")

# Divider
st.markdown("---")

# Table of Results
st.subheader("🧾 User Fraud Scores")
st.dataframe(data[['Fraud_Probability', 'Fraud_Label'] + list(data.columns[:-2])])

# Plot distribution
st.subheader("📊 Fraud Probability Distribution")
fig, ax = plt.subplots()
sns.histplot(data['Fraud_Probability'], kde=True, bins=20)
st.pyplot(fig)

# Optional: Search/Select a User for Explainability
st.subheader("🔍 Explain Individual User Prediction")
selected_index = st.selectbox('Select User Index', data.index)

selected_user = data.iloc[[selected_index]].drop(['Fraud_Probability', 'Fraud_Label'], axis=1)

# SHAP Explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(selected_user)

st.write("Feature impact on Fraud Prediction:")
st.set_option('deprecation.showPyplotGlobalUse', False)
shap.force_plot(explainer.expected_value[1], shap_values[1], selected_user, matplotlib=True)
