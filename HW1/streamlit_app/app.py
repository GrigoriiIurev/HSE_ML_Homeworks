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

eda_mode = st.selectbox(
    "Источник данных для EDA:",
    ["Учебные данные train", "Учебные данные test", "Загрузить свой CSV-файл"]
)

if eda_mode == "Учебные данные train":
    df_eda = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
elif eda_mode == "Учебные данные test":
    df_eda = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')
else:    
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
        st.stop()
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

st.header("2. Моделирование")

model_name = st.selectbox(
    "Выберите модель:",
    [
        "1. Линейная регрессия без обработки",
        "2. Линейная регрессия с scaler",
        "3. Lasso c scaler",
        "4. Lasso c scaler и грид-сёрч",
        "5. ElasticNet c scaler и грид-сёрч",
        "6. Ridge c scaler, категориальными данными и грид-сёрч",
        "7. Ridge - лучшая модель"
    ]
)

MODEL_MAP = {
    "1. Линейная регрессия без обработки": "linear_raw",
    "2. Линейная регрессия с scaler": "linear_scaled",
    "3. Lasso c scaler": "lasso_simple",
    "4. Lasso c scaler и грид-сёрч": "lasso_grid",
    "5. ElasticNet c scaler и грид-сёрч": "elasticnet_grid",
    "6. Ridge c scaler, категориальными данными и грид-сёрч": "ridge_grid",
    "7. Ridge - лучшая модель": "ridge_bonus"
}

model_key = MODEL_MAP[model_name]
model_obj = load_model(model_key)

model          = model_obj["model"]
scaler         = model_obj.get("scaler")
encoder        = model_obj.get("encoder")
feature_order  = model_obj["feature_order"]
num_cols       = model_obj.get("num_cols")
ohe_cols       = model_obj.get("ohe_cols")
is_log_target  = model_obj.get("target_log", False)

uploaded_pred = st.file_uploader("Загрузите CSV для моделирования", type=["csv"], key="model")
if uploaded_pred:
    df_pred = pd.read_csv(uploaded_pred)
    if model_key in ["linear_raw", "linear_scaled", "lasso_simple", "lasso_grid", "elasticnet_grid"]:
        fe_pred = FeatureEngineer(mode="base")
    elif model_key == "ridge_grid":
        fe_pred = FeatureEngineer(mode="medium")
    elif model_key == "ridge_bonus":
        fe_pred = FeatureEngineer(mode="full")

    df_pred_fe = fe_pred.transform(df_pred)
    st.write(f"{num_cols}")
    st.dataframe(df_pred_fe.head())
    if encoder is None:
        X = df_pred_fe[feature_order].copy()
        if scaler is not None:
            X[num_cols] = scaler.transform(X[num_cols])
    else:
        numeric_part = df_pred_fe[num_cols]
        cat_part     = df_pred_fe[ohe_cols]
        if scaler is not None:
            numeric_scaled = scaler.transform(numeric_part)
            numeric_scaled = pd.DataFrame(numeric_scaled, columns=num_cols)
        else:
            numeric_scaled = numeric_part.copy()
        ohe_encoded = encoder.transform(cat_part)
        ohe_cols_final = encoder.get_feature_names_out(ohe_cols)
        ohe_encoded = pd.DataFrame(ohe_encoded, columns=ohe_cols_final)

        X = pd.concat([numeric_scaled, ohe_encoded], axis=1)

        X = X[feature_order]

    st.dataframe(X.head())
    y_pred = model.predict(X)

    if is_log_target:
        y_pred = np.exp(y_pred)

    st.subheader("🔮 Предсказания")
    result = df_pred.copy()
    result["predicted_price"] = y_pred

    st.dataframe(result.head())

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Скачать результат",
        csv,
        file_name="predictions.csv",
        mime="text/csv",
    )