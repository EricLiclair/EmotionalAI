import pickle
import pandas as pd

class EmotionClassifier:
    def __init__(self, model_pickle_file_name:str) -> None:
        """Initializes the class and loads a pickel file as the model"""
        self.loaded_model = self._load_model_from_pickle(
            model_pickle_file_name=model_pickle_file_name)


    def _load_model_from_pickle(self, model_pickle_file_name):
        """Model loading method"""
        with open(model_pickle_file_name, 'rb') as model_pickle:
            return pickle.load(model_pickle)


    def get_loaded_model(self):
        return self.loaded_model
    

    def predict(self, X: pd.DataFrame):
        loaded_model = self.get_loaded_model()
        prediction = loaded_model.predict(X)
        return prediction


model_pickle_file_name = "model.sav"
model = EmotionClassifier(model_pickle_file_name=model_pickle_file_name)

X = pd.read_csv("data/test_data.csv")
X = X.drop(columns=X.columns[0], axis=1)
y_result = pd.read_csv("data/test_result.csv")
y = model.predict(X=X)
