from sklearn.metrics import classification_report

def generate_classification_report(true_labels, predicted_labels):
    """Gera e retorna o relatório de classificação."""
    return classification_report(true_labels, predicted_labels)
