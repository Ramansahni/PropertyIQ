import pandas as pd

def preprocess_furnishing(df):
    """
    Cleans furnishing type in place/returns copy for predictions and training.
    """
    df = df.copy()
    df['furnishing_type'] = df['furnishing_type'].replace({0.0: 'unfurnished', 1.0: 'semifurnished', 2.0: 'furnished'})
    return df
