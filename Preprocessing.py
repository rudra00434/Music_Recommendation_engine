import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.dropna()

    # Normalize numerical features
    scaler = StandardScaler()
    num_cols = ['tempo', 'energy', 'loudness', 'danceability']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df
