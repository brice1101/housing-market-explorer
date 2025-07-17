import pandas as pd
import os

def load_data(data_path: str, target_col: str):
    """Load and clean the dataset from a gzipped CSV."""
    df = pd.read_csv(data_path, compression='gzip')
    df = df.dropna(subset=target_col)
    return df

def split_by_year(df: pd.DataFrame, target_col: str, split_year: int):
    """Split dataset into train/test using a year threshold."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    train_mask = X['Year'] < split_year
    test_mask = X['Year'] >= split_year

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    return X_train, X_test, y_train, y_test
