# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Evitar uso de Tkinter no Matplotlib
import matplotlib
matplotlib.use('Agg')  # Backend para evitar problemas com tkinter

# Carregando o arquivo CSV
# Certifique-se de que o arquivo 'creditcard.csv' está na raiz do projeto
df = pd.read_csv('creditcard.csv')

# Visualizando as primeiras linhas do dataset
print(df.head())

# Verificando o tamanho do dataset
print(f"Dimensão do dataset: {df.shape}")

# Verificando valores ausentes
print(f"Valores ausentes por coluna:\n{df.isnull().sum()}")

# Distribuição das classes (0: Normal, 1: Fraude)
fraud_check = df['Class'].value_counts(sort=True)
fraud_check.plot(kind='bar', rot=0, color='r')
plt.title("Distribuição de Transações Normais e Fraudes")
plt.xlabel("Classe")
plt.ylabel("Frequência")
labels = ['Normal', 'Fraude']
plt.xticks(range(2), labels)
plt.savefig('class_distribution.png')  # Salva o gráfico como imagem
plt.close()

# Separando as transações fraudulentas e normais
fraud_transactions = df[df['Class'] == 1]
normal_transactions = df[df['Class'] == 0]

# Exibindo o número de transações fraudulentas e normais
print(f"Transações fraudulentas: {fraud_transactions.shape[0]}")
print(f"Transações normais: {normal_transactions.shape[0]}")

# Estatísticas descritivas para o valor das transações
print("Estatísticas - Transações fraudulentas:")
print(fraud_transactions['Amount'].describe())

print("Estatísticas - Transações normais:")
print(normal_transactions['Amount'].describe())

# Visualização da distribuição do valor das transações
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
sns.histplot(fraud_transactions['Amount'], bins=50, ax=ax1, color='red', kde=True)
ax1.set_title('Distribuição do Valor - Fraude')
sns.histplot(normal_transactions['Amount'], bins=50, ax=ax2, color='blue', kde=True)
ax2.set_title('Distribuição do Valor - Normal')
plt.xlabel("Valor da Transação")
plt.savefig('transaction_amount_distribution.png')  # Salva o gráfico como imagem
plt.close()

# Aplicação de Isolation Forest para detecção de anomalias
isolation_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df['Anomaly_Score'] = isolation_forest.fit_predict(df.drop('Class', axis=1))

# Exibindo relatório de classificação baseado no modelo
print("Relatório de Classificação - Isolation Forest:")
print(classification_report(df['Class'], df['Anomaly_Score'] == -1))

# Aplicação de One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.01)
df['SVM_Score'] = oc_svm.fit_predict(df.drop('Class', axis=1))

# Exibindo relatório de classificação para SVM
print("Relatório de Classificação - One-Class SVM:")
print(classification_report(df['Class'], df['SVM_Score'] == -1))
