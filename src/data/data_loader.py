import pandas as pd

def load_data(file_path):
    """Carrega os dados do arquivo CSV."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset carregado com sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas.")
        return df
    except Exception as e:
        raise FileNotFoundError(f"Erro ao carregar o arquivo: {e}")
