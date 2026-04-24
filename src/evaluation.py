import numpy as np
import pandas as pd


PROXY_PCT = 0.02  # top/bottom 2% of each variable-city treated as proxy event


def proxy_labels(df, variable, pct=PROXY_PCT):
    # label top and bottom tails per city as proxy events
    df = df.copy()
    label_col = f'proxy_{variable}'
    df[label_col] = False

    for city, sub in df.groupby('CITY'):
        if variable == 'PRCP':
            # PRCP is one-sided (very wet matters, negative values dont exist)
            hi = sub[variable].quantile(1 - pct)
            mask = df.index.isin(sub.index) & (df[variable] >= hi)
        else:
            lo = sub[variable].quantile(pct / 2)
            hi = sub[variable].quantile(1 - pct / 2)
            mask = df.index.isin(sub.index) & ((df[variable] <= lo) | (df[variable] >= hi))
        df.loc[mask, label_col] = True

    return df


def precision_recall_at_flag(flag_series, label_series):
    flag = flag_series.fillna(False).astype(bool)
    label = label_series.fillna(False).astype(bool)
    tp = int((flag & label).sum())
    fp = int((flag & ~label).sum())
    fn = int((~flag & label).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        'flagged': int(flag.sum()),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_methods(df, variables=('TMAX', 'TMIN', 'PRCP')):
    rows = []
    df = df.copy()

    for var in variables:
        df = proxy_labels(df, var)

    for city in sorted(df['CITY'].unique()):
        sub = df[df['CITY'] == city]
        for var in variables:
            label = sub[f'proxy_{var}']

            for method, col in [
                ('Z-score', f'anomaly_z_{var}'),
                ('MAD', f'anomaly_mad_{var}'),
                ('IsolationForest', 'anomaly_if')
            ]:
                metrics = precision_recall_at_flag(sub[col], label)
                metrics.update({'city': city, 'variable': var, 'method': method})
                rows.append(metrics)

    return pd.DataFrame(rows)[
        ['city', 'variable', 'method', 'flagged', 'tp', 'fp', 'fn',
         'precision', 'recall', 'f1']
    ]


def agreement_matrix(df, variable):
    # jaccard index pairwise between methods for one variable across full set
    methods = {
        'Z-score': df[f'anomaly_z_{variable}'].fillna(False).astype(bool),
        'MAD': df[f'anomaly_mad_{variable}'].fillna(False).astype(bool),
        'IsolationForest': df['anomaly_if'].fillna(False).astype(bool)
    }
    names = list(methods.keys())
    mat = pd.DataFrame(np.zeros((len(names), len(names))), index=names, columns=names)
    for a in names:
        for b in names:
            inter = (methods[a] & methods[b]).sum()
            union = (methods[a] | methods[b]).sum()
            mat.loc[a, b] = inter / union if union > 0 else 0.0
    return mat


def stability_over_time(df, variable, freq='YE'):
    # anomaly rate per year per method - lets us see if detectors are stable
    out = []
    flag_cols = {
        'Z-score': f'anomaly_z_{variable}',
        'MAD': f'anomaly_mad_{variable}',
        'IsolationForest': 'anomaly_if'
    }
    tmp = df.copy()
    tmp['year'] = tmp['DATE'].dt.year
    for method, col in flag_cols.items():
        rate = tmp.groupby(['CITY', 'year'])[col].mean().reset_index()
        rate['method'] = method
        rate = rate.rename(columns={col: 'anomaly_rate'})
        out.append(rate)
    return pd.concat(out, ignore_index=True)


def top_anomalies(df, variable, method='Z-score', top_n=10):
    score_cols = {
        'Z-score': f'z_{variable}',
        'MAD': f'mad_score_{variable}',
        'IsolationForest': 'if_score'
    }
    col = score_cols[method]
    out = df.copy()
    out['abs_score'] = out[col].abs()
    cols = ['DATE', 'CITY', variable, col, 'abs_score']
    return (out.sort_values('abs_score', ascending=False)[cols]
            .head(top_n).reset_index(drop=True))
