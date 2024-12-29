# app.py

import streamlit as st
from fraud_detection import CreditCardFraudDetection, load_data
from config import DATA_PATH

def main():
    st.title("Detecção de Fraudes em Cartões de Crédito")
    
    # Carregar os dados diretamente usando a função estática
    df = load_data(DATA_PATH)
    
    # Instanciando a classe
    fraud_detector = CreditCardFraudDetection(DATA_PATH)
    fraud_detector.df = df
    
    # Exibindo as primeiras linhas do dataset
    st.subheader("Primeiras Linhas do Dataset")
    st.write(fraud_detector.df.head())

    # Exibindo relatório de distribuição de classes
    st.subheader("Distribuição das Classes")
    st.write(fraud_detector.generate_report())

    # Exibindo contagem de transações
    st.subheader("Contagem de Transações Fraudulentas e Normais")
    fraud_count, normal_count = fraud_detector.transaction_count()
    st.write(f"Transações fraudulentas: {fraud_count}")
    st.write(f"Transações normais: {normal_count}")

    # Exibindo as estatísticas descritivas
    st.subheader("Estatísticas Descritivas")
    fraud_stats, normal_stats = fraud_detector.descriptive_statistics()
    st.write("Estatísticas - Transações fraudulentas")
    st.write(fraud_stats)
    st.write("Estatísticas - Transações normais")
    st.write(normal_stats)

    # Exibindo o gráfico de distribuição de classes
    st.subheader("Gráfico de Distribuição de Classes")
    fig = fraud_detector.plot_class_distribution()
    st.pyplot(fig)

    # Exibindo o gráfico de distribuição de valores de transações
    st.subheader("Gráfico de Distribuição do Valor das Transações")
    fig = fraud_detector.plot_transaction_amount_distribution()
    st.pyplot(fig)

    # Exibindo o relatório de detecção de anomalias (Isolation Forest)
    st.subheader("Relatório de Detecção de Anomalias - Isolation Forest")
    st.write(fraud_detector.detect_anomalies_isolation_forest())

    # Exibindo o relatório de detecção de anomalias (One-Class SVM)
    st.subheader("Relatório de Detecção de Anomalias - One-Class SVM")
    st.write(fraud_detector.detect_anomalies_ocsvm())

if __name__ == "__main__":
    main()
