# app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

# Set page config
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Load model
model = pickle.load(open('XGBmodel-3.pkl', 'rb'))

# Helper function: Preprocess uploaded raw data
def preprocess_data(raw_data):
    df = raw_data.copy()

    # Convert dates
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df['Return_Date'] = pd.to_datetime(df['Return_Date'], errors='coerce')

    # Handle missing Return_Date
    df['Return_Date'].fillna(pd.Timestamp('1970-01-01'), inplace=True)

    # New features
    df['Is_Returned'] = np.where(df['Return_Status'] == 'Returned', 1, 0)
    df['Days_to_Return'] = (df['Return_Date'] - df['Order_Date']).dt.days
    df['Days_to_Return'] = df['Days_to_Return'].fillna(0)
    df['Days_to_Return_Corrected'] = np.where(df['Days_to_Return'] < 0, 0, df['Days_to_Return'])

    df['Late_Return_Flag'] = np.where(df['Days_to_Return_Corrected'] > 30, 1, 0)
    df['Suspicious_Reason_Flag'] = np.where(df['Return_Reason'].isnull(), 1, 0)
    df['Fast_Return_Flag'] = np.where((df['Is_Returned'] == 1) & (df['Days_to_Return_Corrected'] < 3), 1, 0)
    df['Suspicious_Score'] = df['Late_Return_Flag'] + df['Suspicious_Reason_Flag'] + df['Fast_Return_Flag']

    # Aggregate user stats
    user_agg = df.groupby('User_ID').agg({
        'Order_ID': 'count',
        'Is_Returned': 'sum',
        'Product_Price': ['mean', 'sum']
    })
    user_agg.columns = ['User_Total_Orders', 'User_Total_Returns', 'User_Avg_Order_Value', 'User_Total_Spent']
    user_agg = user_agg.reset_index()

    df = df.merge(user_agg, on='User_ID', how='left')
    df['User_Return_Rate'] = df['User_Total_Returns'] / df['User_Total_Orders']
    df['High_Returner_Flag'] = np.where(df['User_Return_Rate'] > 0.5, 1, 0)

    final_features = [
        'Product_Price', 'Order_Quantity', 'Discount_Applied',
        'Days_to_Return_Corrected', 'Late_Return_Flag', 'Suspicious_Reason_Flag',
        'Fast_Return_Flag', 'Suspicious_Score',
        'User_Total_Orders', 'User_Total_Returns', 'User_Return_Rate',
        'User_Avg_Order_Value', 'User_Total_Spent', 'High_Returner_Flag'
    ]
    return df[final_features]

# App Title
st.title("🛡️ Fraudulent User Detection Dashboard")

# Sidebar
st.sidebar.header('Upload User Data')
uploaded_file = st.sidebar.file_uploader("Upload your user dataset (CSV)", type=["csv"])

# Load or simulate data
if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
    data = preprocess_data(raw_data)
else:
    st.sidebar.write("Using simulated data.")
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

# Predict fraud
pred_probs = model.predict_proba(data)[:, 1]
data['Fraud_Probability'] = pred_probs
data['Fraud_Label'] = np.where(data['Fraud_Probability'] > 0.5, 'Fraudulent', 'Normal')

# KPIs
total_users = len(data)
fraudulent_users = sum(data['Fraud_Label'] == 'Fraudulent')
fraud_rate = fraudulent_users / total_users * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Users", total_users)
col2.metric("Fraudulent Users", fraudulent_users)
col3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")

st.markdown("---")

# Fraud Scores Table
st.subheader("🧾 User Fraud Scores")
st.dataframe(data[['Fraud_Probability', 'Fraud_Label'] + list(data.columns[:-2])])

# Probability Distribution
st.subheader("📊 Fraud Probability Distribution")
fig, ax = plt.subplots()
sns.histplot(data['Fraud_Probability'], kde=True, bins=20)
st.pyplot(fig)

# Explainability
st.subheader("🔍 Explain Individual User Prediction")
selected_index = st.selectbox('Select User Index', data.index)
selected_user = data.iloc[[selected_index]].drop(['Fraud_Probability', 'Fraud_Label'], axis=1)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(selected_user)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_df = pd.DataFrame({
    'feature': selected_user.columns,
    'shap_value': shap_values.flatten()
})
shap_df['abs_shap'] = shap_df['shap_value'].abs()
shap_df = shap_df.sort_values('abs_shap', ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(shap_df['feature'], shap_df['shap_value'])
ax.set_xlabel("Impact on Fraud Prediction")
ax.set_title("Top Features Influencing Prediction")
st.pyplot(fig)
