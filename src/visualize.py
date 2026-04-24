import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns


sns.set_theme(style='whitegrid')
CITY_COLORS = {'NYC': '#1f77b4', 'Chicago': '#d62728', 'LA': '#ff7f0e'}


def plot_architecture(out_dir):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')

    boxes = [
        (0.3, 1.8, 1.7, 1.4, 'NOAA CDO\nraw CSV\n(11,190 rows)', '#eaf0ff'),
        (2.4, 1.8, 1.7, 1.4, 'Cleaning\n(filter cities,\ndrop NaN/dup)', '#eaf0ff'),
        (4.5, 1.8, 1.7, 1.4, 'Feature\nengineering\n(seasonal,\nrolling)', '#eaf0ff'),
        (6.6, 3.3, 1.7, 1.2, 'Seasonal\nz-score', '#fff4e6'),
        (6.6, 1.8, 1.7, 1.2, 'MAD', '#fff4e6'),
        (6.6, 0.3, 1.7, 1.2, 'Isolation\nForest', '#fff4e6'),
        (8.7, 1.8, 1.1, 1.4, 'Anomaly\nflags +\nscores', '#eaf0ff'),
    ]
    for (x, y, w, h, t, fc) in boxes:
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04', linewidth=1.2,
                           edgecolor='#333', facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=9.2)

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                                     mutation_scale=14, linewidth=1.2, color='#333'))

    arr(2.0, 2.5, 2.4, 2.5)
    arr(4.1, 2.5, 4.5, 2.5)
    arr(6.2, 2.7, 6.6, 3.85)
    arr(6.2, 2.5, 6.6, 2.4)
    arr(6.2, 2.3, 6.6, 0.95)
    arr(8.3, 3.85, 8.7, 2.7)
    arr(8.3, 2.4, 8.7, 2.5)
    arr(8.3, 0.95, 8.7, 2.3)

    ax.text(5, 4.6, 'System architecture', ha='center', fontsize=13, fontweight='bold')
    return _save(fig, out_dir, 'architecture.png')


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_temp_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x='CITY', y='TMAX', ax=ax,
                order=['Chicago', 'NYC', 'LA'],
                palette=CITY_COLORS)
    ax.set_title('TMAX distribution by city')
    ax.set_ylabel('TMAX (F)')
    return _save(fig, out_dir, 'eda_tmax_boxplot.png')


def plot_monthly_avg(df, out_dir):
    monthly = df.groupby(['CITY', 'month'])['TMAX'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=monthly, x='month', y='TMAX', hue='CITY', ax=ax,
                 palette=CITY_COLORS, marker='o')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_title('Average monthly TMAX by city')
    ax.set_ylabel('Average TMAX (F)')
    return _save(fig, out_dir, 'eda_monthly_avg.png')


def plot_correlation(df, out_dir):
    corr = df[['TMAX', 'TMIN', 'PRCP', 'temp_range']].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature correlation matrix')
    return _save(fig, out_dir, 'eda_correlation.png')


def plot_anomaly_timeseries(df, city, variable, method, out_dir):
    flag_cols = {
        'Z-score': f'anomaly_z_{variable}',
        'MAD': f'anomaly_mad_{variable}',
        'IsolationForest': 'anomaly_if'
    }
    sub = df[df['CITY'] == city].sort_values('DATE')
    flag = sub[flag_cols[method]].fillna(False)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(sub['DATE'], sub[variable], color='steelblue', linewidth=0.8, alpha=0.8,
            label=variable)
    anom = sub[flag]
    ax.scatter(anom['DATE'], anom[variable], color='crimson', s=22,
               label=f'{method} anomaly', zorder=3)
    ax.set_title(f'{city} - {variable} - {method} flagged anomalies')
    ax.set_ylabel(variable)
    ax.set_xlabel('Date')
    ax.legend(loc='upper right')
    fname = f'anom_{city}_{variable}_{method.replace("-", "").replace(" ", "")}.png'
    return _save(fig, out_dir, fname)


def plot_agreement_heatmap(mat, variable, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(mat, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1, ax=ax)
    ax.set_title(f'Detector agreement (Jaccard) - {variable}')
    fname = f'agreement_{variable}.png'
    return _save(fig, out_dir, fname)


def plot_precision_bars(metrics_df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=metrics_df, x='variable', y='precision',
        hue='method', ax=ax
    )
    ax.set_title('Precision vs extreme-percentile proxy labels')
    ax.set_ylim(0, 1)
    return _save(fig, out_dir, 'eval_precision_bars.png')


def plot_stability(stab_df, variable, out_dir):
    sub = stab_df.copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=sub, x='year', y='anomaly_rate', hue='method',
                 style='CITY', ax=ax, markers=True)
    ax.set_title(f'Anomaly rate per year - {variable}')
    ax.set_ylabel('Flagged fraction')
    fname = f'stability_{variable}.png'
    return _save(fig, out_dir, fname)
