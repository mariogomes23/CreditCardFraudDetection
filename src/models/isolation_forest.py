from sklearn.ensemble import IsolationForest

class IsolationForestModel:
    def __init__(self):
        self.model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

    def train_and_predict(self, df):
        """Treina o modelo e retorna previsões."""
        features = df.drop('Class', axis=1)
        return self.model.fit_predict(features)
