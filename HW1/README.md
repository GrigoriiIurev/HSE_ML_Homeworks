# Структура репозитория

В репозитории находятся:

1. **`AI_HW1_Regression_with_inference_pro.ipynb`**
   Это основной файл с домашним заданием.  
   В нём расположен весь код: предобработка данных, построение моделей, обучение, инференс и остальные необходимые шаги.

2. **`result.md`**
   Этот файл содержит все выводы и ответы, которые требуются по домашнему заданию.  

3. Папки с моделями:
   - **`linear_regression_raw/`** — обычная линейная регрессия без масштабирования.  
   - **`linear_regression_scaled/`** — линейная регрессия со scaler.  
   - **`lasso_simple/`** - модель Lasso, обученную без GridSearch
   - **`lasso_grid/`** - модель Lasso после GridSearch
   - **`elasticnet_grid/`** - модель ElasticNet-модель после GridSearch
   - **`ridge_grid/`** — Ridge после GridSearch.  
   - **`ridge_bonus/`** — финальная лучшая Ridge-модель (самая точная), с расширенным FE.

4. Streamlit-приложение:
   **`streamlit_app/`** - папка с рабочим веб-приложением

5. Feature Engineering для Streamlit:
   **`feature_engineering/`** - папка с классом **`FeatureEngineer`**, используемым Streamlit-приложением