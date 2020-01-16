import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import random

if __name__=="__main__":
    data = []

    for i in range(10000):
        r = random.random()
        data.append({'ActorId': 'john', 'level': 5, 'approved': 0 if r < 0.9 else 1})

        v = random.random()
        data.append({'ActorId': 'mary', 'level': 5, 'approved': 1 if v < 0.9 else 0})

    df = pd.DataFrame(data)
    df.ActorId = pd.Categorical(pd.factorize(df.ActorId)[0])
    df.level = pd.Categorical(pd.factorize(df.level)[0])
    df.approved = pd.Categorical(pd.factorize(df.approved)[0])


    # Split data
    outputs = df['approved']
    inputs = df[['ActorId', 'level']]

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.4,
                                                        random_state=23)

    # Random forest model
    rf = RandomForestRegressor()
    pipeline = Pipeline([("regressor", rf)])
    pipeline.fit(X_train, y_train)

    print('Model trained!')

    filename_p = 'model.sav'
    print('Saving model in %s' % filename_p)
    joblib.dump(pipeline, filename_p)
    print('Model saved!')

