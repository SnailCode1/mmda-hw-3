import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"
TEST_PATH = APP_DIR / "test.csv"
CONFIG_PATH = APP_DIR / "config.json"

st.set_page_config(page_title="Оценка качества вина", page_icon="🍷", layout="wide")


@st.cache_resource
def load_model(path: str, file_version: int):
    return joblib.load(path)


@st.cache_data
def load_test(path: str, file_version: int) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def test_proba(_model, X: pd.DataFrame, model_version: int) -> np.ndarray:
    return _model.predict_proba(X)[:, 1]


def get_threshold_default() -> float:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    threshold = float(config["threshold"])
    return threshold


def get_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "Precision": metrics.precision_score(y_true, pred, zero_division=0),
        "Recall": metrics.recall_score(y_true, pred, zero_division=0),
        "F1": metrics.f1_score(y_true, pred, zero_division=0),
    }


def number_step(low: float, high: float) -> float:
    span = high - low
    if span < 0.1:
        return 0.0001
    if span < 2:
        return 0.01
    if span < 50:
        return 0.1
    return 1.0

model_version = MODEL_PATH.stat().st_mtime_ns

model = load_model(str(MODEL_PATH), model_version)
test = load_test(str(TEST_PATH), TEST_PATH.stat().st_mtime_ns)
default_threshold = get_threshold_default()

feature_names = list(model.feature_names_in_)

X_test = test[feature_names]
y_test = test["target"].to_numpy()

p_test = test_proba(model, X_test, model_version)

if "threshold" not in st.session_state:
    st.session_state.threshold = default_threshold

st.sidebar.header("Порог классификации")
threshold = st.sidebar.slider(
    "Минимальная вероятность хорошего вина",
    min_value=0.0,
    max_value=1.0,
    step=0.001,
    key="threshold",
)
st.sidebar.caption(f"Порог по умолчанию из config.json: {default_threshold:.3f}")
st.sidebar.info(
    "Порог ниже => больше вин в подборке (обычно выше полнота)."
)
st.sidebar.info(
    "Порог выше => меньше вин в подборке (обычно выше точность)."
)

st.title("Оценка качества вина")
st.caption(
    "Логистическая регрессия. Хорошее вино - экспертная оценка не ниже 7. "
)

tab_one, tab_metrics, tab_batch, tab_weights = st.tabs(
    ["Одно вино", "Модельные метрики", "CSV-предсказания", "Коэффициенты модели"]
)


with tab_one:
    st.subheader("Оценка одного вина")
    st.write("Введите характеристики. Диапазоны и медианы взяты из тестовой выборки.")
    columns = st.columns(3)
    values = {}
    numeric_features = [f for f in feature_names if f != "is_red"]

    for index, feature in enumerate(numeric_features):
        s = X_test[feature]
        low, high, median = float(s.min()), float(s.max()), float(s.median())
        with columns[index % 3]:
            values[feature] = st.number_input(
                feature.replace("_", " "),
                min_value=low,
                max_value=high,
                value=median,
                step=number_step(low, high),
                format="%.5f",
                key=f"input_{feature}",
                disabled=low == high,
            )

    values["is_red"] = st.radio("Тип вина", [0, 1], format_func=lambda x: "Красное" if x else "Белое", horizontal=True)
    wine = pd.DataFrame([{name: values[name] for name in feature_names}])
    probability = float(model.predict_proba(wine)[0, 1])
    prediction = int(probability >= threshold)

    left, right = st.columns(2)
    left.metric("Вероятность хорошего вина", f"{probability:.1%}")
    right.metric("Выбранный порог", f"{threshold:.1%}")
    if prediction:
        st.success("Решние модели: рекомендовать вино (класс 1).")
    else:
        st.warning("Решение модели: НЕ рекомендовать вино (класс 0).")


with tab_metrics:
    st.subheader("Метрики на тестовой выборке")
    current_pred = (p_test >= threshold).astype(int)
    scores = get_metrics(y_test, p_test, threshold)
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{scores['Precision']:.3f}")
    col2.metric("Recall", f"{scores['Recall']:.3f}")
    col3.metric("F1", f"{scores['F1']:.3f}")

    st.markdown("#### Матрица ошибок")
    matrix = metrics.confusion_matrix(y_test, current_pred, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(figsize=(5.2, 3.7))
    metrics.ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=["Не хорошее", "Хорошее"]
    ).plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_xlabel("Предсказанный класс")
    ax_cm.set_ylabel("Истинный класс")
    fig_cm.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    tn, fp, fn, tp = matrix.ravel()
    fpr_at_threshold = fp / (fp + tn) if (fp + tn) else 0.0
    recall_at_threshold = scores["Recall"]

    fpr, tpr, _ = metrics.roc_curve(y_test, p_test)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, p_test)
    roc_auc = metrics.roc_auc_score(y_test, p_test)
    ap = metrics.average_precision_score(y_test, p_test)

    st.markdown("#### ROC и Precision–Recall")
    plot_left, plot_right = st.columns(2)

    with plot_left:
        fig_roc, ax_roc = plt.subplots(figsize=(6.2, 4.3))
        ax_roc.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], "--", color="gray", label="Случайная модель")
        ax_roc.scatter(fpr_at_threshold, recall_at_threshold, s=75, zorder=5, label=f"Порог {threshold:.3f}")
        ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC-кривая", xlim=(0, 1), ylim=(0, 1.02))
        ax_roc.legend(fontsize="small")
        ax_roc.grid(alpha=0.25)
        fig_roc.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    with plot_right:
        precision_at_threshold = scores["Precision"] if (tp + fp) else 1.0
        fig_pr, ax_pr = plt.subplots(figsize=(6.2, 4.3))
        ax_pr.plot(pr_recall, pr_precision, label=f"Average Precision = {ap:.3f}")
        ax_pr.axhline(y_test.mean(), ls="--", color="gray", label="Случайная модель")
        ax_pr.scatter(recall_at_threshold, precision_at_threshold, s=75, zorder=5, label=f"Порог {threshold:.3f}")
        ax_pr.set(xlabel="Recall", ylabel="Precision", title="PR-кривая", xlim=(0, 1), ylim=(0, 1.02))
        ax_pr.legend(fontsize="small")
        ax_pr.grid(alpha=0.25)
        fig_pr.tight_layout()
        st.pyplot(fig_pr)
        plt.close(fig_pr)

    if (tp + fp) == 0:
        st.caption("При отсутствии положительных прогнозов приняли, что Precision = 0. Точка PR-кривой построена по Precision=1 при Recall=0.")
    st.info(
        "Повышение порога делает подборку более строгой и рекомендованных вин становится меньше, "
        "Recall не увеличивается, а Precision часто растёт. "
        "Понижение порога увеличивает число рекомендаций и Recall, но может понизить Precision."
    )


with tab_batch:
    st.subheader("Пакетная оценка из CSV")
    st.write("Загрузите CSV с 12 признаками вина (без target). Используется порог из сайдбара.")
    st.download_button(
        "Скачать шаблон CSV",
        data=X_test.head(5).to_csv(index=False).encode("utf-8-sig"),
        file_name="wine_template.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader("Выберите CSV-файл", type="csv")

    if uploaded_file is not None:
        try:
            table = pd.read_csv(uploaded_file)
            missing_columns = [name for name in feature_names if name not in table.columns]
            if missing_columns:
                st.error("В CSV не хватает столбцов: " + ", ".join(missing_columns))
            elif table.empty:
                st.error("CSV-файл не содержит строк с данными.")
            else:
                if "target" in table.columns:
                    st.warning("Столбец target был проигнорирован, т. к. для предсказаний он не нужен.")
                data = table.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
                if data.isna().any().any() or not np.isfinite(data.to_numpy()).all():
                    st.error("В файле есть пустые или нечисловые значения в признаках.")
                elif not data["is_red"].isin([0, 1]).all():
                    st.error("Признак is_red должен содержать только 0 (белое) и 1 (красное).")
                else:
                    probabilities = model.predict_proba(data)[:, 1]
                    predictions = (probabilities >= threshold).astype(int)
                    output = table.drop(columns=["target"], errors="ignore").copy()
                    output["probability_good"] = probabilities
                    output["prediction"] = predictions
                    output["recommendation"] = np.where(predictions == 1, "Рекомендовать", "Не рекомендовать")
                    st.dataframe(output, use_container_width=True)
                    st.download_button(
                        "Скачать результаты CSV",
                        data=output.to_csv(index=False).encode("utf-8-sig"),
                        file_name="wine_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Не удалось обработать CSV: {exc}")


with tab_weights:
    st.subheader("Что выучила модель")
    clf = model.steps[-1][1]
    if not hasattr(clf, "coef_"):
        st.error("У сохранённой модели отсутствуют коэффициенты coef_.")
    else:
        coefs = pd.Series(clf.coef_[0], index=feature_names).sort_values()
        fig_coef, ax_coef = plt.subplots(figsize=(9, 6))
        colors = ["#b45309" if value < 0 else "#2563eb" for value in coefs]
        ax_coef.barh(coefs.index.str.replace("_", " "), coefs.values, color=colors)
        ax_coef.axvline(0, color="gray", linewidth=0.9)
        ax_coef.set_xlabel("Коэффициент после стандартизации признаков")
        ax_coef.set_title("Влияние признаков на логарифм отношения шансов")
        fig_coef.tight_layout()
        st.pyplot(fig_coef)
        plt.close(fig_coef)

        top_positive = ", ".join(coefs.nlargest(3).index.str.replace("_", " "))
        top_negative = ", ".join(coefs.nsmallest(3).index.str.replace("_", " "))
        st.write(f"Сильнее всего за хорошее вино: {top_positive}.")
        st.write(f"Сильнее всего против: {top_negative}.")
