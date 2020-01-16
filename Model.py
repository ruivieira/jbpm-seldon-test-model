from sklearn.externals import joblib

class Model(object):

    def __init__(self):
        print("Initializing.")
        print("Loading model.")
        self.model = joblib.load('model.sav')

    def predict(self,X,features_names):
        return self.model.predict_proba(X)