import pandas as pd


CITIES = ['Chicago', 'NYC', 'LA']
KEEP_COLS = ['DATE', 'CITY', 'TMAX', 'TMIN', 'PRCP']


def get_city(name):
    name = str(name).upper()
    if 'NY' in name:
        return 'NYC'
    elif 'CHICAGO' in name:
        return 'Chicago'
    elif 'LOS ANGELES' in name:
        return 'LA'
    else:
        return 'Other'


def load_raw(path):
    df = pd.read_csv(path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    return df


def clean(df):
    df = df.copy()
    df['CITY'] = df['NAME'].apply(get_city)
    df = df[df['CITY'].isin(CITIES)]
    df = df[KEEP_COLS]
    df = df.dropna(subset=['TMAX', 'TMIN', 'PRCP'])
    df = df.drop_duplicates()
    df = df.sort_values(['CITY', 'DATE']).reset_index(drop=True)
    return df


def save(df, path):
    df.to_csv(path, index=False)
