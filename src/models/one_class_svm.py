from sklearn.svm import OneClassSVM

class OneClassSVMModel:
    def __init__(self):
        self.model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.01)

    def train_and_predict(self, df):
        """Treina o modelo e retorna previsões."""
        features = df.drop('Class', axis=1)
        return self.model.fit_predict(features)
