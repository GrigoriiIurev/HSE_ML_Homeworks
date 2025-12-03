import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys
import plotly.express as px
import phik
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

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

input_mode = st.radio(
    "Способ ввода данных:",
    ["Загрузить CSV", "Ввести данные вручную"]
)

if input_mode == "Загрузить CSV":
    uploaded_pred = st.file_uploader("Загрузите CSV для моделирования", type=["csv"], key="model")

    if uploaded_pred:
        df_pred = pd.read_csv(uploaded_pred)
        process_data = True
    else:
        process_data = False
else:
    process_data = False

if input_mode == "Ввести данные вручную":
    with st.form("manual_input"):
        st.subheader("📝 Введите данные автомобиля")

        name = st.text_input("Name", "Maruti Swift Dzire VDI")
        year = st.number_input("Year", min_value=1980, max_value=2025, value=2014)
        km_driven = st.number_input("km_driven", min_value=0, max_value=5000000, value=70000)
        fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG", "LPG"])
        seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner',
                                        'Fourth & Above Owner', 'Test Drive Car'])
        mileage = st.text_input("Mileage", "18.0 kmpl")
        engine = st.text_input("Engine", "1248 CC")
        max_power = st.text_input("Max Power", "82 bhp")
        torque = st.text_input("Torque", "113 Nm @ 4500 rpm")
        seats = st.number_input("Seats", min_value=2, max_value=10, value=5)

        submitted = st.form_submit_button("Предсказать")

    if submitted:
        df_pred = pd.DataFrame([{
            "name": name,
            "year": year,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "torque": torque,
            "seats": seats
        }])

        process_data = True

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

model = model_obj["model"]
scaler = model_obj.get("scaler")
encoder = model_obj.get("encoder")
feature_order = model_obj["feature_order"]
num_cols = model_obj.get("num_cols")
ohe_cols = model_obj.get("ohe_cols")
is_log_target = model_obj.get("target_log", False)

if process_data:
    if model_key in ["linear_raw", "linear_scaled", "lasso_simple", "lasso_grid", "elasticnet_grid"]:
        fe_pred = FeatureEngineer(mode="base")
    elif model_key == "ridge_grid":
        fe_pred = FeatureEngineer(mode="medium")
    elif model_key == "ridge_bonus":
        fe_pred = FeatureEngineer(mode="full")

    df_pred_fe = fe_pred.transform(df_pred)
    if encoder is None:
        X = df_pred_fe[feature_order].copy()
        if scaler is not None:
            X[num_cols] = scaler.transform(X[num_cols])

    else:
        numeric_part = df_pred_fe[num_cols]
        cat_part     = df_pred_fe[ohe_cols].astype(str)

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

    y_pred = model.predict(X)
    if is_log_target:
        y_pred = np.exp(y_pred)

    st.subheader("Предсказания")
    result = df_pred.copy()
    result["predicted_price"] = y_pred
    st.dataframe(result.head())

    true_col_candidates = ["selling_price", "price", "y", "target"]
    true_col = next((c for c in true_col_candidates if c in df_pred.columns), None)

    if true_col is not None:
        y_true = df_pred[true_col].values
        y_pred_corrected = result["predicted_price"].values

        mae = mean_absolute_error(y_true, y_pred_corrected)
        mse = mean_squared_error(y_true, y_pred_corrected)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred_corrected)

        st.subheader("Метрики модели")
        st.markdown(f"""
        **MSE:** {mse:,.2f}  
        **RMSE:** {rmse:,.2f}  
        **R2:** {r2:.4f}
        """)
    else:
        st.info("В данных нет настоящей цены и метрики не посчитать.")

if hasattr(model, "coef_"):
    st.subheader("Веса модели")

    coef = np.asarray(model.coef_).ravel()
    if len(coef) != len(feature_order):
        st.warning(
            f"Размерность coef_ ({len(coef)}) не совпадает с количеством признаков "
            f"feature_order ({len(feature_order)})."
        )
    else:
        coef_df = pd.DataFrame({
            "feature": feature_order,
            "weight": coef
        })

        coef_df = coef_df.sort_values("weight", ascending=False)

        height = max(6, 0.15 * len(coef_df))

        fig, ax = plt.subplots(figsize=(8, height))

        sns.barplot(
            data=coef_df,
            y="feature",
            x="weight",
            order=coef_df["feature"],
            orient="h",
            ax=ax
        )

        ax.set_title("Веса линейной модели)")
        ax.set_xlabel("коэффициент")
        ax.set_ylabel("признак")

        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(coef_df)