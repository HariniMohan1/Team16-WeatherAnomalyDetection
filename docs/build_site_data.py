import os
import json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DATA_DIR = os.path.join(ROOT, 'data')
TABLE_DIR = os.path.join(ROOT, 'outputs', 'tables')
OUT = os.path.join(HERE, 'data')
os.makedirs(OUT, exist_ok=True)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'model_output_full.csv'), parse_dates=['DATE'])

    series = {}
    top_flags = {}
    for city in ['NYC', 'Chicago', 'LA']:
        series[city] = {}
        top_flags[city] = {}
        sub = df[df['CITY'] == city].sort_values('DATE').reset_index(drop=True)
        for var in ['TMAX', 'TMIN', 'PRCP']:
            dates = sub['DATE'].dt.strftime('%Y-%m-%d').tolist()
            values = sub[var].round(2).tolist()
            z = sub[f'anomaly_z_{var}'].fillna(False).astype(bool).tolist()
            mad = sub[f'anomaly_mad_{var}'].fillna(False).astype(bool).tolist()
            iforest = sub['anomaly_if'].fillna(False).astype(bool).tolist()
            series[city][var] = {
                'dates': dates,
                'values': values,
                'z': z,
                'mad': mad,
                'if': iforest,
            }

            zcol = f'z_{var}'
            if zcol in sub.columns:
                top = (sub[sub[f'anomaly_z_{var}'].fillna(False)]
                       .assign(abs_z=lambda d: d[zcol].abs())
                       .sort_values('abs_z', ascending=False)
                       .head(10)[['DATE', var, zcol]]
                       .rename(columns={var: 'value', zcol: 'z'}))
                top['DATE'] = top['DATE'].dt.strftime('%Y-%m-%d')
                top_flags[city][var] = top.to_dict(orient='records')
            else:
                top_flags[city][var] = []

    metrics = pd.read_csv(os.path.join(TABLE_DIR, 'metrics_summary.csv'))
    metrics_out = metrics.round(3).to_dict(orient='records')

    agreement = {}
    for var in ['TMAX', 'TMIN', 'PRCP']:
        a = pd.read_csv(os.path.join(TABLE_DIR, f'agreement_{var}.csv'), index_col=0)
        agreement[var] = {
            'rows': a.index.tolist(),
            'cols': a.columns.tolist(),
            'values': a.round(3).values.tolist()
        }

    stability = {}
    for var in ['TMAX', 'TMIN', 'PRCP']:
        s = pd.read_csv(os.path.join(TABLE_DIR, f'stability_{var}.csv'))
        stability[var] = s.round(4).to_dict(orient='records')

    payload = {
        'series': series,
        'topFlags': top_flags,
        'metrics': metrics_out,
        'agreement': agreement,
        'stability': stability,
    }

    path = os.path.join(OUT, 'payload.json')
    with open(path, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    kb = os.path.getsize(path) / 1024
    print(f'wrote {path} ({kb:.1f} KB)')


if __name__ == '__main__':
    main()
