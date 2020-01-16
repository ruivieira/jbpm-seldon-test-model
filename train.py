import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import random

if __name__=="__main__":
    data = []

    prices = np.random.normal(loc=1500.0, scale=40.0, size=3000)
    print(prices)
    for price in prices:
        data.append({'brand': 'Lenovo', 'price': price, 'approved': 1})

    prices = np.random.normal(loc=2500.0, scale=40.0, size=3000)
    for price in prices:
        data.append({'brand': 'Apple', 'price': price, 'approved': 1})
        data.append({'brand': 'Lenovo', 'price': price, 'approved': 0})

    df = pd.DataFrame(data)
    df.brand = pd.Categorical(pd.factorize(df.brand)[0])
    df.approved = pd.Categorical(pd.factorize(df.approved)[0])


    # Split data
    outputs = df['approved']
    inputs = df[['brand', 'price']]

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.4,
                                                        random_state=23)

    # Random forest model
    rf = RandomForestClassifier()
    pipeline = Pipeline([("classifier", rf)])
    pipeline.fit(X_train, y_train)

    print('Model trained!')

    filename_p = 'model.sav'
    print('Saving model in %s' % filename_p)
    joblib.dump(pipeline, filename_p)
    print('Model saved!')

    model = joblib.load('model.sav')