import pandas as pd

def split_na(df, col):
    """Splits a dataframe by whether a value is missing in a column"""
    dfT = select.df[~df[col].isna()])
    dfF = select.df[df[col].isna()])
    return dfT,dfF

    
