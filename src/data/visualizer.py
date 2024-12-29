import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df, save_path):
    """Plota a distribuição das classes."""
    fraud_check = df['Class'].value_counts(sort=True)
    fraud_check.plot(kind='bar', rot=0, color='r')
    plt.title("Distribuição de Transações Normais e Fraudes")
    plt.xlabel("Classe")
    plt.ylabel("Frequência")
    plt.savefig(f"{save_path}class_distribution.png")
    plt.close()

def plot_transaction_distribution(df, save_path):
    """Plota a distribuição do valor das transações."""
    fraud_transactions = df[df['Class'] == 1]
    normal_transactions = df[df['Class'] == 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    sns.histplot(fraud_transactions['Amount'], bins=50, ax=ax1, color='red', kde=True)
    ax1.set_title('Distribuição do Valor - Fraude')
    sns.histplot(normal_transactions['Amount'], bins=50, ax=ax2, color='blue', kde=True)
    ax2.set_title('Distribuição do Valor - Normal')
    plt.xlabel("Valor da Transação")
    plt.savefig(f"{save_path}transaction_amount_distribution.png")
    plt.close()
