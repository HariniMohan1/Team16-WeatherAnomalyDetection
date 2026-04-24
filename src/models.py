import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


Z_THRESHOLD = 2.0
MAD_THRESHOLD = 3.5
IF_CONTAMINATION = 0.02
RANDOM_STATE = 42


def fit_seasonal_baseline(train, variable):
    # seasonal mean/std per city + day of year, learned on training data only
    g = train.groupby(['CITY', 'dayofyear'])[variable]
    baseline = pd.DataFrame({
        'seasonal_mean': g.mean(),
        'seasonal_std': g.std()
    }).reset_index()
    return baseline


def apply_zscore(df, baseline, variable, threshold=Z_THRESHOLD):
    df = df.merge(baseline, on=['CITY', 'dayofyear'], how='left',
                  suffixes=('', f'_{variable}_base'))
    # guard against zero std on rare days
    std = df['seasonal_std'].replace(0, np.nan)
    df[f'z_{variable}'] = (df[variable] - df['seasonal_mean']) / std
    df[f'anomaly_z_{variable}'] = df[f'z_{variable}'].abs() > threshold
    df = df.rename(columns={
        'seasonal_mean': f'seasonal_mean_{variable}',
        'seasonal_std': f'seasonal_std_{variable}'
    })
    return df


def fit_mad_baseline(train, variable):
    # median + MAD per city + day of year. robust to outliers and skew
    g = train.groupby(['CITY', 'dayofyear'])[variable]
    med = g.median().rename('seasonal_median')

    def mad_fn(x):
        m = np.median(x)
        return np.median(np.abs(x - m))

    mad = g.apply(mad_fn).rename('seasonal_mad')
    baseline = pd.concat([med, mad], axis=1).reset_index()
    return baseline


def apply_mad(df, baseline, variable, threshold=MAD_THRESHOLD):
    df = df.merge(baseline, on=['CITY', 'dayofyear'], how='left')
    # 0.6745 converts MAD to normal-consistent sigma
    scale = 0.6745
    denom = df['seasonal_mad'].replace(0, np.nan)
    df[f'mad_score_{variable}'] = scale * (df[variable] - df['seasonal_median']) / denom
    df[f'anomaly_mad_{variable}'] = df[f'mad_score_{variable}'].abs() > threshold
    df = df.rename(columns={
        'seasonal_median': f'seasonal_median_{variable}',
        'seasonal_mad': f'seasonal_mad_{variable}'
    })
    return df


def fit_isolation_forest(train, feature_cols, contamination=IF_CONTAMINATION):
    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    model.fit(train[feature_cols])
    return model


def apply_isolation_forest(df, model, feature_cols):
    df = df.copy()
    df['if_score'] = -model.score_samples(df[feature_cols])  # higher = more anomalous
    df['anomaly_if'] = model.predict(df[feature_cols]) == -1
    return df


def run_all_detectors(train, test, variables=('TMAX', 'TMIN', 'PRCP'),
                      if_features=('TMAX', 'TMIN', 'PRCP', 'temp_range', 'rolling_mean')):
    scored = test.copy()

    for var in variables:
        z_base = fit_seasonal_baseline(train, var)
        scored = apply_zscore(scored, z_base, var)

        mad_base = fit_mad_baseline(train, var)
        scored = apply_mad(scored, mad_base, var)

    if_model = fit_isolation_forest(train, list(if_features))
    scored = apply_isolation_forest(scored, if_model, list(if_features))

    return scored, if_model
