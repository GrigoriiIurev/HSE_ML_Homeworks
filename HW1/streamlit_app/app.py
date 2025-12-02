import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys
import plotly.express as px
import phik

base_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(base_dir, "..", "feature_engineering"))
sys.path.append(parent_dir)
from MultiModelPipeline import FeatureEngineer


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


st.title("Часть 5 | Создание интерактивного приложения на Streamlit")

st.header("1. Exploratory Data Analysis")

uploaded_eda = st.file_uploader("Загрузите CSV для EDA", type=["csv"], key="eda")

if uploaded_eda:
    df_eda = pd.read_csv(uploaded_eda)
    st.write("Первые строки:")
    st.dataframe(df_eda.head())

    fe = FeatureEngineer(mode="EDA")
    df_eda = fe.transform(df_eda)

    numeric_cols = df_eda.select_dtypes(include=["int", "float"]).columns
st.markdown(
    """
    <div style="
        background-color:#f0f4ff;
        border-left:6px solid #1a73e8;
        padding:12px 18px;
        border-radius:4px;
        font-size:16px;
        color:#0b2545;
        margin-top:20px;
        ">
        🔍 <b>Важно:</b> колонка <code>torque</code> была автоматически разобрана на два признака:
        <ul>
            <li><b>torque</b> — очищенное значение момента</li>
            <li><b>max_torque_rpm</b> — максимальные обороты</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
st.subheader("📈 Интерактивное распределение признаков")

if uploaded_eda:
    # выбираем тип графика
    plot_type = st.selectbox(
        "Тип графика:",
        ["Scatterplot", "Histogram", "KDE Plot"]
    )

    # выбор колонок
    numeric_cols = df_eda.select_dtypes(include=["int", "float"]).columns
    
    x_col = st.selectbox("X:", numeric_cols)
    y_col = st.selectbox("Y:", numeric_cols)
    if plot_type == "Scatterplot":
        if x_col == y_col:
            st.error("❌ Нельзя выбрать одинаковые признаки для X и Y. Выбери разные колонки.")
        else:
            fig = px.scatter(
                df_eda,
                x=x_col,
                y=y_col,
                title=f"Scatter: {x_col} vs {y_col}",
                hover_data=df_eda.columns,
                opacity=0.7,
                trendline="ols"
            )

            # уменьшаем точки
            fig.update_traces(
                marker=dict(size=6),
                selector=dict(mode="markers")
            )

            # окрашиваем линию регрессии
            for trace in fig.data:
                if trace.mode == "lines":
                    trace.line.color = "red"
                    trace.line.width = 3

    elif plot_type == "Histogram":
        fig = px.histogram(
            df_eda,
            x=x_col,
            nbins=40,
            title=f"Histogram: {x_col}",
            opacity=0.8
        )

    elif plot_type == "KDE Plot":
        fig = px.histogram(
            df_eda,
            x=x_col,
            nbins=120,
            histnorm="probability density",
            marginal="box",
            opacity=0.6,
            title=f"KDE Density: {x_col}"
        )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Pairplot (Scatter Matrix)")

if uploaded_eda:

    pairplot_cols = st.multiselect(
        "Выберите признаки для pairplot:",
        numeric_cols,
        default=list(numeric_cols[:4])
    )

    if len(pairplot_cols) > 1:
        if st.button("Построить pairplot"):
            fig = px.scatter_matrix(
                df_eda[pairplot_cols],
                dimensions=pairplot_cols,
                title="Scatter Matrix (Pairplot)",
                height=800,
                width=800
            )
            fig.update_traces(diagonal_visible=True, showupperhalf=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Выбери хотя бы 2 числовых признака.")

st.subheader("📌 Корреляционные матрицы")

numeric_df = df_eda.select_dtypes(include=["int", "float"])

corr_type = st.selectbox(
    "Метрика корреляции:",
    ["Пирсон", "Спирмен", "Phik"]
)

if corr_type == "Пирсон":
    corr = numeric_df.corr(method="pearson")

elif corr_type == "Спирмен":
    corr = numeric_df.corr(method="spearman")

elif corr_type == "Phik":
    corr = numeric_df.phik_matrix(interval_cols=numeric_df.columns.tolist())

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu",
    title=f"Корреляционная матрица ({corr_type})"
)

st.plotly_chart(fig, use_container_width=True)

st.header("3. Моделирование")
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
