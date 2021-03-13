import pandas as pd

def split_na(df, col):
    """Splits a dataframe by whether a value is missing in a column"""
    isna = df[col].isna()
    return df[~isna], df[isna]


    
