import os
import sys
import argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)

import data_cleaning as dc
import feature_engineering as fe
import models as md
import evaluation as ev
import visualize as vz


def main(raw_path, data_dir, fig_dir, table_dir, split_date):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    print('Loading raw dataset from', raw_path)
    raw = dc.load_raw(raw_path)
    print('Raw rows:', len(raw))

    clean = dc.clean(raw)
    print('Clean rows:', len(clean))

    feat = fe.add_features(clean)
    clean_path = os.path.join(data_dir, 'clean_weather_data.csv')
    dc.save(feat, clean_path)
    print('Saved clean features to', clean_path)

    # EDA plots off the full feature frame
    vz.plot_architecture(fig_dir)
    vz.plot_temp_distribution(feat, fig_dir)
    vz.plot_monthly_avg(feat, fig_dir)
    vz.plot_correlation(feat, fig_dir)

    train, test = fe.time_split(feat, split_date=split_date)
    print(f'Train rows: {len(train)} (<{split_date})')
    print(f'Test rows:  {len(test)}  (>={split_date})')

    scored, _if_model = md.run_all_detectors(train, test)
    model_out_path = os.path.join(data_dir, 'model_output_data.csv')
    scored.to_csv(model_out_path, index=False)
    print('Saved model outputs to', model_out_path)

    # also score the full feature frame so figures cover all 10 years visually
    full_scored, _ = md.run_all_detectors(train, feat)
    full_out_path = os.path.join(data_dir, 'model_output_full.csv')
    full_scored.to_csv(full_out_path, index=False)
    print('Saved full scored frame to', full_out_path)

    # evaluation on test window only
    metrics = ev.evaluate_methods(scored)
    metrics_path = os.path.join(table_dir, 'metrics_by_city_variable_method.csv')
    metrics.to_csv(metrics_path, index=False)
    print('Saved metrics to', metrics_path)

    summary = (metrics.groupby(['variable', 'method'])
               [['flagged', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1']]
               .mean().round(3).reset_index())
    summary_path = os.path.join(table_dir, 'metrics_summary.csv')
    summary.to_csv(summary_path, index=False)
    print('Saved summary to', summary_path)

    vz.plot_precision_bars(metrics, fig_dir)

    for var in ['TMAX', 'TMIN', 'PRCP']:
        mat = ev.agreement_matrix(scored, var)
        mat_path = os.path.join(table_dir, f'agreement_{var}.csv')
        mat.to_csv(mat_path)
        vz.plot_agreement_heatmap(mat, var, fig_dir)

        stab = ev.stability_over_time(full_scored, var)
        stab_path = os.path.join(table_dir, f'stability_{var}.csv')
        stab.to_csv(stab_path, index=False)
        vz.plot_stability(stab, var, fig_dir)

        for city in ['NYC', 'Chicago', 'LA']:
            for method in ['Z-score', 'MAD', 'IsolationForest']:
                vz.plot_anomaly_timeseries(full_scored, city, var, method, fig_dir)

        top = ev.top_anomalies(scored, var, method='Z-score', top_n=15)
        top_path = os.path.join(table_dir, f'top_anomalies_{var}_zscore.csv')
        top.to_csv(top_path, index=False)

    print('Done.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default=os.path.join(ROOT, 'data', 'OriginalWeatherDataset.csv'))
    p.add_argument('--data-dir', default=os.path.join(ROOT, 'data'))
    p.add_argument('--fig-dir', default=os.path.join(ROOT, 'outputs', 'figures'))
    p.add_argument('--table-dir', default=os.path.join(ROOT, 'outputs', 'tables'))
    p.add_argument('--split-date', default='2023-01-01')
    args = p.parse_args()
    main(args.raw, args.data_dir, args.fig_dir, args.table_dir, args.split_date)
