def preprocess_data(df):
    """Realiza pré-processamento dos dados."""
    # Exemplo de pré-processamento
    df.dropna(inplace=True)  # Remover valores nulos
    print(f"Dados após pré-processamento: {df.shape[0]} linhas, {df.shape[1]} colunas.")
    return df
