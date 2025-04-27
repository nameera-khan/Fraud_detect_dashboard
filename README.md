# 🛡️ Fraudulent User Detection Dashboard

This repository contains an end-to-end Machine Learning project for detecting fraudulent refund abuse users in an e-commerce setting, featuring a fully interactive dashboard built with Streamlit.


## 📚 Project Overview

- **Objective**: Predict which users are likely to commit refund abuse based on their purchase and return behavior.
- **Model**: XGBoost Classifier
- **Explainability**: SHAP Values
- **Deployment**: Streamlit App with Tabbed Navigation

The project includes:
- Data cleaning and validation
- Feature engineering inspired by real-world fraud scenarios
- Model training and evaluation
- Explainable AI integration
- User-friendly web dashboard


## 📊 App Features

- Upload your own transaction datasets
- Automatic data preprocessing and feature creation
- Fraud probability prediction for each user
- Interactive KPIs (Fraud Rate, Total Users, Detected Frauds)
- Probability distribution visualization
- Individual user fraud explainability using SHAP values


## 🛠️ Project Structure

```
fraud_detect_dashboard/
├── app.py                # Main Streamlit app
├── XGBmodel-3.pkl        # Trained fraud detection model
├── requirements.txt      # Python package dependencies
├── README.md             # Project documentation
```

---

## 🔥 Key Learnings

- Handling dirty and inconsistent data (Return Date anomalies)
- Creating domain-driven features to boost model performance
- Building interpretable ML systems with SHAP
- Deploying fast and user-friendly Streamlit dashboards


## ⚡ Requirements
- Python 3.9+
- Streamlit
- Pandas
- Numpy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- SHAP

(Already listed in `requirements.txt`)


## 📢 Limitations
- Dataset is synthetic: Model achieves very high accuracy (perfect), which is unlikely on real-world messy data.
- Production deployment would require handling user feedback loops, model recalibration, and threshold optimization.


## 📋 Future Enhancements
- Connect to live databases
- Add authentication to the app
- Deploy to Streamlit Cloud or AWS
- Introduce anomaly detection layer for unseen fraud tactics


## 🙌 Acknowledgements
- Dataset inspired by synthetic e-commerce returns behavior.
- Streamlit and SHAP libraries for rapid app development and explainability.



