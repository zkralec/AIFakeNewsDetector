import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def get_label_distribution(df, label_col='label'):
    return df[label_col].value_counts()
