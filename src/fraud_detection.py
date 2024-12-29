# fraud_detection.py

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

class CreditCardFraudDetection:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def generate_report(self):
        """Gera os relatórios de classe"""
        if 'Class' in self.df.columns:
            fraud_check = self.df['Class'].value_counts(sort=True)
            return fraud_check
        else:
            return "A coluna 'Class' não foi encontrada. Verifique os dados."

    def transaction_count(self):
        """Conta as transações fraudulentas e normais"""
        fraud_transactions = self.df[self.df['Class'] == 1]
        normal_transactions = self.df[self.df['Class'] == 0]
        return fraud_transactions.shape[0], normal_transactions.shape[0]

    def descriptive_statistics(self):
        """Exibe as estatísticas descritivas das transações"""
        fraud_transactions = self.df[self.df['Class'] == 1]
        normal_transactions = self.df[self.df['Class'] == 0]
        
        return fraud_transactions['Amount'].describe(), normal_transactions['Amount'].describe()

    def plot_class_distribution(self):
        """Plota a distribuição das classes"""
        fraud_check = self.generate_report()
        fig, ax = plt.subplots()
        fraud_check.plot(kind='bar', rot=0, color='r', ax=ax)
        ax.set_title("Distribuição de Transações Normais e Fraudes")
        ax.set_xlabel("Classe")
        ax.set_ylabel("Frequência")
        ax.set_xticklabels(['Normal', 'Fraude'])
        return fig

    def plot_transaction_amount_distribution(self):
        """Plota a distribuição do valor das transações"""
        fraud_transactions = self.df[self.df['Class'] == 1]
        normal_transactions = self.df[self.df['Class'] == 0]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        sns.histplot(fraud_transactions['Amount'], bins=50, ax=ax1, color='red', kde=True)
        ax1.set_title('Distribuição do Valor - Fraude')
        sns.histplot(normal_transactions['Amount'], bins=50, ax=ax2, color='blue', kde=True)
        ax2.set_title('Distribuição do Valor - Normal')
        plt.xlabel("Valor da Transação")
        return fig

    def detect_anomalies_isolation_forest(self):
        """Aplica o Isolation Forest para detectar anomalias"""
        isolation_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        self.df['Anomaly_Score'] = isolation_forest.fit_predict(self.df.drop('Class', axis=1))
        return classification_report(self.df['Class'], self.df['Anomaly_Score'] == -1)

    def detect_anomalies_ocsvm(self):
        """Aplica o One-Class SVM para detectar anomalias"""
        oc_svm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.01)
        self.df['SVM_Score'] = oc_svm.fit_predict(self.df.drop('Class', axis=1))
        return classification_report(self.df['Class'], self.df['SVM_Score'] == -1)

# Função para carregar os dados, que não depende de self
@st.cache_data
def load_data(data_path):
    """Carregar o dataset e limpar colunas"""
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()  # Limpar espaços nos nomes das colunas
    return df
