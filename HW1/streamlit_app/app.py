import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

@st.cache_resource
def load_model(model_name):
    base_dir = os.path.dirname(__file__)

    models = {
        "linear_raw": ("..", "linear_regression_raw", "all_in_one.pkl"),
        "linear_scaled": ("..", "linear_regression_scaled", "all_in_one.pkl"),
        "lasso_simple": ("..", "lasso_simple", "all_in_one.pkl"),
        "lasso_grid": ("..", "lasso_grid", "all_in_one.pkl"),
        "ridge_grid": ("..", "ridge_grid", "all_in_one.pkl"),
        "elasticnet_grid": ("..", "elasticnet_grid", "all_in_one.pkl"),
        "ridge_bonus": ("..", "ridge_bonus", "all_in_one.pkl"),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    path_parts = models[model_name]
    model_path = os.path.join(base_dir, *path_parts)

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    return obj


st.title("Car Price Predictor — Linear Models")

st.header("1. Exploratory Data Analysis")

uploaded_eda = st.file_uploader("Загрузите CSV для EDA", type=["csv"], key="eda")

if uploaded_eda:
    df_eda = pd.read_csv(uploaded_eda)
    st.write("Первые строки:")
    st.dataframe(df_eda.head())

    numeric_cols = df_eda.select_dtypes(include=["int", "float"]).columns

    st.subheader("Гистограммы числовых признаков")

    for col in numeric_cols:
        fig, ax = plt.subplots()
        ax.hist(df_eda[col].dropna(), bins=30, color="skyblue", edgecolor="black")
        ax.set_title(col)
        st.pyplot(fig)

model_name = st.selectbox(
    "Выберите модель:",
    [
        "linear_raw",
        "linear_scaled",
        "lasso_simple",
        "lasso_grid",
        "ridge_grid",
        "elasticnet_grid",
        "ridge_bonus"
    ]
)

model_obj = load_model(model_name)
model = model_obj["model"]
scaler = model_obj.get("scaler", None)
encoder = model_obj.get("encoder", None)
feature_order = model_obj["feature_order"]
is_log_target = model_obj.get("target_log", False)
