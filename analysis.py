import pandas as pd
import scipy.stats as ss
from itertools import combinations


def prepare_table(df, period):
    """Aggregate records by selected period."""
    if period == 'Неделя':
        df['period'] = df['date'] - pd.to_timedelta(df['date'].dt.weekday, unit='D')
    elif period == 'Месяц':
        df['period'] = df['date'].values.astype('datetime64[M]')
    else:
        df['period'] = df['date']

    mean_pattern = 'SleepAnalysis|Weight|BMI|Variability|HRV'
    metric_means = df[df['metric'].str.contains(mean_pattern)]
    metric_sums = df[~df['metric'].str.contains(mean_pattern)]

    pt_sum = metric_sums.pivot_table(index='period', columns='metric', values='value', aggfunc='sum')
    pt_mean = metric_means.pivot_table(index='period', columns='metric', values='value', aggfunc='mean')
    return pd.concat([pt_sum, pt_mean], axis=1).sort_index()


def analyze_pairs(wide_df, p_thr, min_N):
    """Return DataFrame with significant correlations."""
    results = []
    pairs = combinations(wide_df.columns.dropna(), 2)
    for X, Y in pairs:
        series = wide_df[[X, Y]].dropna()
        N = len(series)
        if N < min_N:
            continue
        if series[X].nunique() < 2 or series[Y].nunique() < 2:
            continue
        try:
            res = ss.linregress(series[X], series[Y])
        except ValueError:
            continue
        if res.pvalue > p_thr:
            continue
        results.append({'X_raw': X, 'Y_raw': Y, 'r': res.rvalue, 'p': res.pvalue, 'N': N})
    return pd.DataFrame(results).sort_values('p')


def compute_delta_optx(wide_df, pairs_df):
    """Calculate ΔY and optimal X for each pair."""
    delta_list = []
    optx_list = []
    for _, row in pairs_df.iterrows():
        X = row['X_raw']
        Y = row['Y_raw']
        series = wide_df[[X, Y]].dropna()
        dy = series[Y].diff().dropna().mean()
        delta_list.append(dy)
        thresh = 0.95 * series[Y].max()
        xs = series[X][series[Y] >= thresh]
        optx_list.append(xs.min() if not xs.empty else None)
    pairs_df['ΔY'] = delta_list
    pairs_df['OptX'] = optx_list
    return pairs_df


def pretty(name):
    base = name
    base = base.replace('HKCategoryTypeIdentifier', '')
    base = base.replace('HKQuantityTypeIdentifier', '')
    return ''.join([' ' + c if c.isupper() else c for c in base]).strip()
