import pandas as pd


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Normalize values between 0 and 1"""
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val)

def apply_weights(df: pd.DataFrame, weights_dict: dict[str, float]) -> pd.DataFrame:
    """Apply weights to normalized columns"""
    for var_name, weight in weights_dict.items():
        df[f"{var_name}_w"] = df[var_name] * weight
    return df

def normalize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalize multiple columns"""
    for col in columns:
        df[f"{col}_norm"] = min_max_normalize(df[col])
    return df 