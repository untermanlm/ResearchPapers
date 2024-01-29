import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def preprocess(file: Path, outname: str) -> None:
    '''
    Preprocesses specific DataFrame file and saves to csv
    '''
    # Read csv
    df =  pd.read_csv(file)
    # Channel column removed
    df = df.drop('Channel', axis=1)
    #  columns Partnered, Mature, and Language become numerical
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object' or df[column].dtype == 'bool':
            df[column] = label_encoder.fit_transform(df[column])
    # Min-Max normalization applied 
    normalized_df = (df-df.min())/(df.max()-df.min())
    # null values removed
    cleared_df = normalized_df.dropna(axis=0)
    # write to csv
    outfile = file.parent / outname
    cleared_df.to_csv(outfile, index=False)
