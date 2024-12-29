# utils.py

import pandas as pd

def load_data(data_path):

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()  # Limpar espaços nos nomes das colunas
    return df
