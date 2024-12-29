
import matplotlib
import sys
import os

from config import Config
from data.data_loader import load_data
from data.pre_processor import preprocess_data
from data.visualizer import plot_class_distribution, plot_transaction_distribution
from models.isolation_forest import IsolationForestModel
from models.one_class_svm import OneClassSVMModel
from models.metrics import generate_classification_report

matplotlib.use('Agg')  # Define o backend para um que não depende de GUI

# Adicionar o diretório 'src' ao caminho do Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    # Carregar dados
    df = load_data(Config.DATA_PATH)

    # Pré-processamento
    df = preprocess_data(df)

    # Visualização
    plot_class_distribution(df, save_path=Config.OUTPUT_PATH)
    plot_transaction_distribution(df, save_path=Config.OUTPUT_PATH)

    # Treinar e avaliar modelos
    isolation_forest_model = IsolationForestModel()
    df['Anomaly_Score'] = isolation_forest_model.train_and_predict(df)

    svm_model = OneClassSVMModel()
    df['SVM_Score'] = svm_model.train_and_predict(df)

    # Relatórios
    print("Relatório de Classificação - Isolation Forest:")
    print(generate_classification_report(df['Class'], df['Anomaly_Score'] == -1))

    print("Relatório de Classificação - One-Class SVM:")
    print(generate_classification_report(df['Class'], df['SVM_Score'] == -1))
