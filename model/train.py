import joblib
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_dataset():
    print("Iris dataset is loaded")
    return datasets.load_iris()

def split_data():
    iris_df = load_dataset()

    X = iris_df.data
    scaler = MinMaxScaler() # added MinMaxScaler
    scaled_X = scaler.fit_transform(X)

    y = iris_df.target
    y = pd.get_dummies(y) # added get_dummies

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.25, random_state=12, shuffle=True)
    return X_train, X_test, y_train, y_test, scaler

def model_fitting(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    print("Model is trained")
    return model

def make_predictions(X_test, trained_model):
    preds = trained_model.predict(X_test)
    return preds

def model_accuracy(y_test, preds):
    accuracy = accuracy_score(y_test, preds)
    print("Model test data accuracy: %s" % accuracy)

def save_model(trained_model, scaler):
    joblib.dump(trained_model, 'trained_models/iris-model.pkl')
    print("Model is saved")
    joblib.dump(scaler, 'trained_models/iris-scaler.pkl') # added also saving the scaler
    print("Scaler is saved")

def train_model():
    X_train, X_test, y_train, y_test, scaler = split_data()
    trained_model = model_fitting(X_train, y_train)
    predictions = make_predictions(X_test, trained_model)
    model_accuracy(y_test, predictions)
    save_model(trained_model, scaler)