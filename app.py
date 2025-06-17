import streamlit as st
import io
import logging
import csv
import pandas as pd
import plotly.express as px
import re

from parser import load_data, get_types
from analysis import (
    prepare_table,
    analyze_pairs,
    compute_delta_optx,
    pretty,
)

logging.basicConfig(level=logging.DEBUG)

# Expand page width
st.set_page_config(layout="wide")

# Sidebar: upload group mapping CSV
mapping_upload = st.sidebar.file_uploader(
    "Загрузите mapping CSV (Metric_to_Group_Mapping.csv)", type="csv", key="mapping"
)
# Load static mapping from uploaded file or fallback
group_mapping = {}
if mapping_upload is not None:
    reader = csv.DictReader(io.StringIO(mapping_upload.getvalue().decode("utf-8")))
    for row in reader:
        group_mapping[row["Metric"]] = row["Group"]
else:
    st.sidebar.warning("Файл mapping CSV не загружен — все метрики попадут в 'other'.")


def get_group(metric_pretty_name):
    return group_mapping.get(metric_pretty_name, "other")


# buffer for in-app logs
log_buffer = io.StringIO()
buffer_handler = logging.StreamHandler(log_buffer)
buffer_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(buffer_handler)

# list for collecting debug messages
log_messages = []

st.title("Health XML: корреляция метрик")

uploaded = st.file_uploader("Загрузите Apple Health XML", type="xml")
if not uploaded:
    st.stop()
file_bytes = uploaded.read()

types = get_types(file_bytes)


st.sidebar.markdown("### Метрика X (день)")
x_type = st.sidebar.selectbox(
    "Ось X",
    types,
    index=(
        types.index("HKQuantityTypeIdentifierStepCount")
        if "HKQuantityTypeIdentifierStepCount" in types
        else 0
    ),
)

st.sidebar.markdown("### Метрика Y (ночь)")
y_type = st.sidebar.selectbox(
    "Ось Y",
    types,
    index=(
        types.index("HKCategoryTypeIdentifierSleepAnalysis")
        if "HKCategoryTypeIdentifierSleepAnalysis" in types
        else 0
    ),
)

# Выбор периода агрегации
period = st.sidebar.selectbox("Гранулярность", ["День", "Неделя", "Месяц"])
# Порог p-value
p_thr = st.sidebar.slider(
    "Порог значимости p-value", min_value=0.01, max_value=0.1, value=0.05, step=0.01
)

# Option to exclude trivial same-group pairs
drop_same = st.sidebar.checkbox("Исключить пары из одной группы", value=True)

# Vectorized workflow
raw_df = load_data(file_bytes)
wide_df = prepare_table(raw_df, period)
# determine min_N threshold: median observations per metric or 10
min_N = max(10, int(wide_df.notna().sum().median()))
pairs_df = analyze_pairs(wide_df, p_thr, min_N)


if not pairs_df.empty:
    df2 = compute_delta_optx(wide_df, pairs_df.copy())
    df2["X"] = df2["X_raw"].map(pretty)
    df2["Y"] = df2["Y_raw"].map(pretty)
    # format p-value to three decimal places

    def fmt_p(p):
        # format p-value to three decimal places
        return f"{p:.3f}"

    df2["p"] = df2["p"].apply(fmt_p)

    if drop_same:
        diff_mask = df2["X"].map(get_group) != df2["Y"].map(get_group)
        df2 = df2[diff_mask].reset_index(drop=True)

    # Поиск в таблице
    search = st.text_input("🔍 Поиск метрик (X или Y)", "")
    if search:
        mask = df2["X"].str.contains(search, case=False) | df2["Y"].str.contains(
            search, case=False
        )
        df_display = df2.loc[mask]
    else:
        df_display = df2
    st.dataframe(df_display[["X", "Y", "r", "p", "ΔY", "OptX", "N"]])

    # Топ межгрупповых связей
    if st.sidebar.checkbox("Показать топ-5 межгрупповых связей по |r|", value=False):
        cross_mask = df2["X"].map(get_group) != df2["Y"].map(get_group)
        cross_df = df2[cross_mask].copy()
        # сортируем по абсолютному r и берём топ-5
        cross_df["abs_r"] = cross_df["r"].abs()
        top5 = cross_df.sort_values("abs_r", ascending=False).head(5)
        st.subheader("Топ-5 связей между разными группами")
        st.dataframe(
            top5[["X", "Y", "r", "p", "N", "ΔY", "OptX"]], use_container_width=True
        )
else:
    st.write("Нет значимых связей для выбранного порога p-value.")

# Interactive description per row
with st.expander("Интерактивное описание"):
    selected_row_index = st.number_input(
        "Введите номер строки для подробного просмотра",
        min_value=0,
        max_value=len(df2) - 1 if not pairs_df.empty else 0,
        step=1,
        value=0,
    )
    if not pairs_df.empty and 0 <= selected_row_index < len(df2):
        row = df2.iloc[selected_row_index]
        st.write(f"**Параметр X:** {row['X']} (оригинал: {row['X_raw']})")
        st.write(f"**Параметр Y:** {row['Y']} (оригинал: {row['Y_raw']})")
        st.write(f"**Коэффициент корреляции (r):** {row['r']:.3f}")
        st.write(f"**p-значение:** {row['p']}")
        st.write(f"**Количество наблюдений (N):** {row['N']}")
        st.write(f"**Среднее изменение ΔY:** {row['ΔY']:.3f}")
        st.write(f"**Оптимальное значение X (где Y >= 95% max):** {row['OptX']}")
        # Plot scatter
        orig_X = row["X_raw"]
        orig_Y = row["Y_raw"]
        series = wide_df[[orig_X, orig_Y]].dropna()
        # Compute and display average and expected benefit
        mean_x = series[orig_X].mean()
        mean_y = series[orig_Y].mean()
        threshold = 0.95 * series[orig_Y].max()
        expected_benefit = threshold - mean_y
        st.write(f"Сейчас ваш средний **{row['X']}** ≈ {mean_x:.1f}.")
        st.write(
            f"При увеличении до **{row['OptX']:.1f}** ваш средний **{row['Y']}** может вырасти примерно на {expected_benefit:.1f}."
        )
        fig = px.scatter(
            series,
            x=orig_X,
            y=orig_Y,
            title=f"График зависимости {row['Y']} от {row['X']}",
            labels={orig_X: row["X"], orig_Y: row["Y"]},
        )
        st.plotly_chart(fig, use_container_width=True)
