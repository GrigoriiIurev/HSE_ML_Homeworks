import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# =========================================
# Загружаем модель
# =========================================

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "ridge_bonus", "all_in_one.pkl")

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    return obj

model_obj = load_model()
model = model_obj["model"]
scaler = model_obj.get("scaler", None)
encoder = model_obj.get("encoder", None)
feature_order = model_obj["feature_order"]
is_log_target = model_obj.get("target_log", False)

st.title("🚗 Car Price Predictor — Linear Models")

st.write("""
Это приложение построено на основе линейных моделей (включая Ridge).
Здесь вы можете:
- посмотреть EDA-графики,
- загрузить CSV и получить предсказания,
- вручную ввести признаки,
- посмотреть коэффициенты модели.
""")


# =========================================
# 1. EDA SECTION
# =========================================
st.header("📊 1. Exploratory Data Analysis")

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


# =========================================
# 2. Prediction
# =========================================
st.header("🎯 2. Предсказание цены автомобиля")

mode = st.radio(
    "Выберите режим предсказания:",
    ("Загрузить CSV", "Ввести признаки вручную")
)

# -------- Функция подготовки данных ------------
def prepare_features(df):
    # строгий порядок признаков
    df = df[feature_order].copy()

    # scaling
    if scaler is not None:
        df[scaler.feature_names_in_] = scaler.transform(df[scaler.feature_names_in_])

    # OHE
    if encoder is not None:
        ohe = encoder.transform(df[encoder.feature_names_in_])
        ohe_df = pd.DataFrame(ohe, columns=encoder.get_feature_names_out(), index=df.index)
        df = df.drop(columns=encoder.feature_names_in_)
        df = pd.concat([df, ohe_df], axis=1)

    # приведение к финальному порядку
    df = df.reindex(columns=feature_order, fill_value=0)

    return df

# ----------- CSV режим -----------
if mode == "Загрузить CSV":
    uploaded_pred = st.file_uploader("Загрузите CSV", type=["csv"], key="csvpred")
    if uploaded_pred:
        df_input = pd.read_csv(uploaded_pred)
        st.dataframe(df_input.head())

        X = prepare_features(df_input)
        y_pred = model.predict(X)

        if is_log_target:
            y_pred = np.expm1(y_pred)

        df_input["prediction"] = y_pred
        st.subheader("Результат")
        st.dataframe(df_input)

        st.download_button(
            "Скачать с предсказаниями",
            data=df_input.to_csv(index=False),
            file_name="predictions.csv"
        )


# ----------- Manual Input -----------
else:
    st.write("Введите значения признаков:")

    input_dict = {}
    for f in feature_order:
        if encoder and f in encoder.feature_names_in_:
            continue  # OHE будет отдельно handled

        val = st.number_input(f"{f}", value=0.0)
        input_dict[f] = val

    df_man = pd.DataFrame([input_dict])

    X = prepare_features(df_man)
    y_pred = model.predict(X)
    if is_log_target:
        y_pred = np.expm1(y_pred)

    st.subheader("Предсказанная цена:")
    st.success(f"💰 {int(y_pred[0]):,} ₹".replace(",", " "))


# =========================================
# 3. Model weights
# =========================================
st.header("⚙️ 3. Веса модели (коэффициенты)")

coefs = model.coef_
coef_df = pd.DataFrame({
    "feature": feature_order,
    "coef": coefs
}).sort_values("coef", ascending=False)

st.dataframe(coef_df)

st.subheader("График важности признаков")
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(coef_df["feature"], coef_df["coef"], color="orange")
ax.set_xlabel("Вес")
ax.set_ylabel("Признак")
ax.set_title("Коэффициенты модели")
st.pyplot(fig)