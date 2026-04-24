import pandas as pd


ROLLING_WINDOW = 7


def add_features(df):
    df = df.copy()
    df['month'] = df['DATE'].dt.month
    df['dayofyear'] = df['DATE'].dt.dayofyear
    df['temp_range'] = df['TMAX'] - df['TMIN']

    # rolling stats per city so NYC values dont contaminate LA and so on
    df['rolling_mean'] = df.groupby('CITY')['TMAX'].transform(
        lambda x: x.rolling(ROLLING_WINDOW).mean()
    )
    df['rolling_std'] = df.groupby('CITY')['TMAX'].transform(
        lambda x: x.rolling(ROLLING_WINDOW).std()
    )
    df['rolling_mean_tmin'] = df.groupby('CITY')['TMIN'].transform(
        lambda x: x.rolling(ROLLING_WINDOW).mean()
    )
    df['rolling_mean_prcp'] = df.groupby('CITY')['PRCP'].transform(
        lambda x: x.rolling(ROLLING_WINDOW).mean()
    )

    df = df.dropna().reset_index(drop=True)
    return df


def time_split(df, split_date='2023-01-01'):
    df = df.copy()
    split_ts = pd.to_datetime(split_date)
    train = df[df['DATE'] < split_ts].reset_index(drop=True)
    test = df[df['DATE'] >= split_ts].reset_index(drop=True)
    return train, test
