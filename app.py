import streamlit as st
from io import BytesIO
import io
from lxml import etree
from datetime import datetime, timedelta
from collections import defaultdict
import scipy.stats as ss
import plotly.express as px
import numpy as np
import logging
import traceback
import statistics
import pandas as pd
import re
logging.basicConfig(level=logging.DEBUG)

import csv

import io

# Expand page width
st.set_page_config(layout="wide")

# Sidebar: upload group mapping CSV
mapping_upload = st.sidebar.file_uploader(
    "Загрузите mapping CSV (Metric_to_Group_Mapping.csv)",
    type="csv", key="mapping"
)
# Load static mapping from uploaded file or fallback
group_mapping = {}
if mapping_upload is not None:
    reader = csv.DictReader(io.StringIO(mapping_upload.getvalue().decode('utf-8')))
    for row in reader:
        group_mapping[row['Metric']] = row['Group']
else:
    st.sidebar.warning("Файл mapping CSV не загружен — все метрики попадут в 'other'.")

def get_group(metric_pretty_name):
    return group_mapping.get(metric_pretty_name, 'other')


# buffer for in-app logs
log_buffer = io.StringIO()
buffer_handler = logging.StreamHandler(log_buffer)
buffer_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(buffer_handler)

# list for collecting debug messages
log_messages = []


# 1. Load raw XML into a DataFrame of (date, metric, value)
def load_data(file_bytes):
    records = []
    for _, elem in etree.iterparse(BytesIO(file_bytes), tag='Record'):
        t = elem.get('type')
        start = elem.get('startDate')
        if not t or not start:
            elem.clear(); continue
        dt = datetime.fromisoformat(start).date()
        if t.endswith('SleepAnalysis'):
            if elem.get('value') != 'HKCategoryValueSleepAnalysisAsleepDeep':
                elem.clear(); continue
            val = (datetime.fromisoformat(elem.get('endDate')) - datetime.fromisoformat(start)).total_seconds() / 60
            records.append({'date': dt, 'metric': t, 'value': val})
            # record bedtime (startDate hour+minute) as a separate metric
            dt_start = datetime.fromisoformat(start)
            bedtime_val = dt_start.hour * 60 + dt_start.minute
            records.append({'date': dt_start.date(), 'metric': 'Bedtime', 'value': bedtime_val})
            # record wake-up time (endDate hour+minute) as a separate metric
            dt_end = datetime.fromisoformat(elem.get('endDate'))
            wake_val = dt_end.hour * 60 + dt_end.minute
            records.append({'date': dt_end.date(), 'metric': 'WakeTime', 'value': wake_val})
        else:
            try:
                val = float(elem.get('value') or 0)
            except:
                val = 0
            records.append({'date': dt, 'metric': t, 'value': val})
        elem.clear()
    return pd.DataFrame(records)

# 2. Prepare pivot table aggregated by period
from pandas import Grouper

def prepare_table(df, period):
    # map period levels
    if period == 'Неделя':
        df['period'] = df['date'] - pd.to_timedelta(df['date'].dt.weekday, unit='D')
    elif period == 'Месяц':
        df['period'] = df['date'].values.astype('datetime64[M]')
    else:
        df['period'] = df['date']
    # metrics to average per period (sleep depths, body stats, variability)
    mean_pattern = 'SleepAnalysis|Weight|BMI|Variability|HRV'
    metric_means = df[df['metric'].str.contains(mean_pattern)]
    metric_sums  = df[~df['metric'].str.contains(mean_pattern)]
    pt_sum = metric_sums.pivot_table(
        index='period', columns='metric', values='value', aggfunc='sum')
    pt_mean = metric_means.pivot_table(
        index='period', columns='metric', values='value', aggfunc='mean')
    return pd.concat([pt_sum, pt_mean], axis=1).sort_index()

# 3. Analyze all pairs vectorized
from itertools import combinations

def analyze_pairs(wide_df, p_thr, min_N):
    results = []
    metrics = wide_df.columns.dropna()
    pairs = list(combinations(metrics, 2))
    progress_bar = st.progress(0, text="Анализируем пары метрик...")
    for idx, (X, Y) in enumerate(pairs):
        series = wide_df[[X, Y]].dropna()
        N = len(series)
        if N < min_N:
            progress_bar.progress((idx + 1) / len(pairs))
            continue
        # skip if no variation in X or Y
        if series[X].nunique() < 2 or series[Y].nunique() < 2:
            progress_bar.progress((idx + 1) / len(pairs))
            continue
        try:
            res = ss.linregress(series[X], series[Y])
        except ValueError:
            progress_bar.progress((idx + 1) / len(pairs))
            continue
        if res.pvalue > p_thr:
            progress_bar.progress((idx + 1) / len(pairs))
            continue
        results.append({'X': X, 'Y': Y, 'r': res.rvalue, 'p': res.pvalue, 'N': N})
        progress_bar.progress((idx + 1) / len(pairs))
    progress_bar.empty()
    return pd.DataFrame(results).sort_values('p')

@st.cache_data(show_spinner=False)
def get_types(file_bytes):
    """
    Возвращает отсортированный список уникальных типов <Record> в файле.
    """
    types = set()
    for _, elem in etree.iterparse(BytesIO(file_bytes), tag='Record'):
        t = elem.get('type')
        if t:
            types.add(t)
        elem.clear()
    return sorted(types)

st.title("Health XML: корреляция метрик")

uploaded = st.file_uploader("Загрузите Apple Health XML", type="xml")
if not uploaded:
    st.stop()
file_bytes = uploaded.read()

types = get_types(file_bytes)


st.sidebar.markdown("### Метрика X (день)")
x_type = st.sidebar.selectbox(
    "Ось X", types,
    index=types.index("HKQuantityTypeIdentifierStepCount") if "HKQuantityTypeIdentifierStepCount" in types else 0
)

st.sidebar.markdown("### Метрика Y (ночь)")
y_type = st.sidebar.selectbox(
    "Ось Y", types,
    index=types.index("HKCategoryTypeIdentifierSleepAnalysis") if "HKCategoryTypeIdentifierSleepAnalysis" in types else 0
)

# Выбор периода агрегации
period = st.sidebar.selectbox('Гранулярность', ['День', 'Неделя', 'Месяц'])
# Порог p-value
p_thr = st.sidebar.slider('Порог значимости p-value', min_value=0.01, max_value=0.1, value=0.05, step=0.01)

# Option to exclude trivial same-group pairs
drop_same = st.sidebar.checkbox('Исключить пары из одной группы', value=True)

# Vectorized workflow
raw_df = load_data(file_bytes)
wide_df = prepare_table(raw_df, period)
# determine min_N threshold: median N across all pairs or 10
all_counts = []
for X, Y in combinations(wide_df.columns, 2):
    all_counts.append(len(wide_df[[X, Y]].dropna()))
min_N = max(10, int(pd.Series(all_counts).median()))
pairs_df = analyze_pairs(wide_df, p_thr, min_N)

# human-readable metric name
def pretty(name):
    # remove HealthKit prefix
    base = re.sub(r'^HK(?:Category|Quantity)TypeIdentifier', '', name)
    # split camel case
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', base).strip()


if not pairs_df.empty:
    df2 = pairs_df.copy()
    # compute ΔY and Оптимальное X for each pair
    delta_list = []
    optx_list = []
    for _, row in df2.iterrows():
        X, Y = row['X'], row['Y']
        # original metric names
        orig_X = pairs_df.loc[_,'X']
        orig_Y = pairs_df.loc[_,'Y']
        series = wide_df[[orig_X, orig_Y]].dropna()
        # ΔY: average period-to-period change
        dy = series[orig_Y].diff().dropna()
        delta_list.append(dy.mean())
        # Оптимальное X: minimal X where Y>=95% of its max
        thresh = 0.95 * series[orig_Y].max()
        xs = series[orig_X][series[orig_Y] >= thresh]
        optx_list.append(xs.min() if not xs.empty else None)
    # assign
    df2['ΔY'] = delta_list
    df2['OptX'] = optx_list

    # preserve original columns before mapping pretty names
    df2['X_raw'] = df2['X']
    df2['Y_raw'] = df2['Y']
    df2['X'] = df2['X'].map(pretty)
    df2['Y'] = df2['Y'].map(pretty)
    # format p-value to three decimal places
    def fmt_p(p):
        # format p-value to three decimal places
        return f"{p:.3f}"
    df2['p'] = df2['p'].apply(fmt_p)

    if drop_same:
        # keep only pairs where groups differ (using pretty names)
        df2 = df2[df2.apply(lambda row: get_group(row['X']) != get_group(row['Y']), axis=1)].reset_index(drop=True)

    # Поиск в таблице
    search = st.text_input('🔍 Поиск метрик (X или Y)', '')
    if search:
        mask = df2['X'].str.contains(search, case=False) | df2['Y'].str.contains(search, case=False)
        df_display = df2.loc[mask]
    else:
        df_display = df2
    st.dataframe(df_display[['X','Y','r','p','ΔY','OptX','N']])

    # Топ межгрупповых связей
    if st.sidebar.checkbox('Показать топ-5 межгрупповых связей по |r|', value=False):
        # фильтруем пары из разных групп по pretty names
        cross_mask = []
        for _, row in df2.iterrows():
            if get_group(row['X']) != get_group(row['Y']):
                cross_mask.append(True)
            else:
                cross_mask.append(False)
        cross_df = df2.loc[cross_mask].copy()
        # сортируем по абсолютному r и берём топ-5
        cross_df['abs_r'] = cross_df['r'].abs()
        top5 = cross_df.sort_values('abs_r', ascending=False).head(5)
        st.subheader('Топ-5 связей между разными группами')
        st.dataframe(top5[['X','Y','r','p','N','ΔY','OptX']], use_container_width=True)
else:
    st.write("Нет значимых связей для выбранного порога p-value.")

# Interactive description per row
with st.expander("Интерактивное описание"):
    selected_row_index = st.number_input(
        "Введите номер строки для подробного просмотра",
        min_value=0,
        max_value=len(df2) - 1 if not pairs_df.empty else 0,
        step=1,
        value=0
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
        orig_X = row['X_raw']
        orig_Y = row['Y_raw']
        series = wide_df[[orig_X, orig_Y]].dropna()
        # Compute and display average and expected benefit
        mean_x = series[orig_X].mean()
        mean_y = series[orig_Y].mean()
        threshold = 0.95 * series[orig_Y].max()
        expected_benefit = threshold - mean_y
        st.write(f"Сейчас ваш средний **{row['X']}** ≈ {mean_x:.1f}.")
        st.write(f"При увеличении до **{row['OptX']:.1f}** ваш средний **{row['Y']}** может вырасти примерно на {expected_benefit:.1f}.")
        fig = px.scatter(series, x=orig_X, y=orig_Y,
                         title=f"График зависимости {row['Y']} от {row['X']}",
                         labels={orig_X: row['X'], orig_Y: row['Y']})
        st.plotly_chart(fig, use_container_width=True)
