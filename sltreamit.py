import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Page config
st.set_page_config(page_title="Mobile Price Predictor", layout="wide")

st.title("📱 Mobile Phone Price Range Prediction System")
st.write("Predict whether a phone is Low, Medium, High or Very High cost.")

# -----------------------------------
# Upload Dataset
# -----------------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload dataset.csv", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded Successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------
    # Feature Engineering
    # -----------------------------------
    df['screen_size'] = df['sc_h'] * df['sc_w']
    df['pixel_density'] = (df['px_height'] * df['px_width']) / df['screen_size']
    df['avg_camera'] = (df['fc'] + df['pc']) / 2
    df['memory_per_core'] = df['int_memory'] / df['n_cores']
    df['power_efficiency'] = df['battery_power'] / df['talk_time']

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    feature_cols = [col for col in df.columns if col != 'price_range']

    X = df[feature_cols]
    y = df['price_range']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------------
    # Train Model
    # -----------------------------------
    st.sidebar.header("Model Settings")
    if st.sidebar.button("Train Model"):

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"Model Trained Successfully! Accuracy: {acc:.2%}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        # Feature Importance
        st.subheader("Top Feature Importances")
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance_df.head(10))

        fig2, ax2 = plt.subplots()
        sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", ax=ax2)
        st.pyplot(fig2)

        # -----------------------------------
        # User Input Section
        # -----------------------------------
        st.subheader("🔍 Predict New Mobile Price")

        user_input = {}

        col1, col2 = st.columns(2)

        for i, feature in enumerate(feature_cols):
            if i % 2 == 0:
                user_input[feature] = col1.number_input(feature, value=float(X[feature].mean()))
            else:
                user_input[feature] = col2.number_input(feature, value=float(X[feature].mean()))

        if st.button("Predict Price Range"):

            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]

            price_map = {
                0: "Low Cost",
                1: "Medium Cost",
                2: "High Cost",
                3: "Very High Cost"
            }

            st.success(f"Predicted Price Range: {price_map[prediction]}")

else:
    st.info("Please upload the dataset to begin.")
