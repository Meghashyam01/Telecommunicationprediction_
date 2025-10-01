# -----------------------------
# Telecom Churn Predictor (v2.0)
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")

# -----------------------------
# Load Data & Model
# -----------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("telecommunications_churn.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset missing! Upload 'telecommunications_churn.csv'.")
        st.stop()

def preprocess_data(df):
    df = df.copy()
    le = LabelEncoder()
    for col in ["voice_mail_plan", "international_plan", "churn"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    if "total_charge" in df.columns:
        df.drop(columns=["total_charge"], inplace=True)
    return df

@st.cache_resource
def load_model():
    try:
        model, scaler, feature_names = pickle.load(open("churn_xgb_model.pkl", "rb"))
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("⚠️ Model missing! Please place 'churn_xgb_model.pkl' in the app directory.")
        st.stop()

# Load once
raw_df = load_data()
processed_df = preprocess_data(raw_df)
model, scaler, feature_names = load_model()

# -----------------------------
# New Top Navigation
# -----------------------------
menu = st.tabs(["🏠 Home", "👤 Customer Lookup", "📂 File Upload & Predictions", "📑 Model Performance"])

# -----------------------------
# Home
# -----------------------------
with menu[0]:
    st.title("🏠 Welcome to the Telecom Churn Predictor")
    st.markdown(
        """
        This app helps telecom companies analyze **customer churn** and make predictions.  
        Use the navigation tabs above to explore different functionalities.  

        ### Features:
        - 👤 Predict churn for a single customer (Customer Lookup)  
        - 📂 Upload CSV and predict churn in bulk  
        - 📑 View model performance metrics  

        **Tech Stack:** Streamlit · Scikit-learn · XGBoost · Pandas
        """
    )

# -----------------------------
# Customer Lookup (Single Prediction)
# -----------------------------
with menu[1]:
    st.header("👤 Customer Lookup")
    st.write("Fill in customer details below to predict churn:")

    input_dict = {}
    for feature in feature_names:
        if feature in ["voice_mail_plan", "international_plan"]:
            input_dict[feature] = 1 if st.radio(f"{feature}:", ["No", "Yes"]) == "Yes" else 0
        else:
            input_dict[feature] = st.number_input(f"{feature}:", value=0.0)

    if st.button("🔮 Run Prediction"):
        input_data = pd.DataFrame([input_dict]).reindex(columns=feature_names, fill_value=0)
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        if pred == 1:
            st.error("⚠️ This customer is likely to **CHURN**!")
        else:
            st.success("✅ This customer will likely **STAY**.")

# -----------------------------
# Bulk Prediction
# -----------------------------
with menu[2]:
    st.header("📂 File Upload & Predictions")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        new_data_enc = preprocess_data(new_data)

        X_new = new_data_enc.reindex(columns=feature_names, fill_value=0)
        X_new_scaled = scaler.transform(X_new)
        preds = model.predict(X_new_scaled)

        new_data["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in preds]
        st.dataframe(new_data.head())

        csv = new_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv,
                           file_name="Churn_Predictions.csv", mime="text/csv")

# -----------------------------
# Model Performance
# -----------------------------
with menu[3]:
    st.header("📑 Model Performance")
    st.write("Evaluating model on available dataset:")

    X = processed_df.drop(columns=["churn"])
    y = processed_df["churn"]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    acc = accuracy_score(y, y_pred)
    st.metric("Model Accuracy", f"{acc:.2f}")

    st.subheader("Classification Report")
    st.write(pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.write(pd.DataFrame(cm, index=["Actual: No Churn", "Actual: Churn"],
                          columns=["Pred: No Churn", "Pred: Churn"]))
