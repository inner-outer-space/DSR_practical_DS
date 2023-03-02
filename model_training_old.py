import pandas as pd
import numpy as np
import pickle as pickle

import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn import tree
from sklearn.metrics import accuracy_score


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

    
def Encoding(data):
    category_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

    # deal witht the space in TotalCharges
    data['TotalCharges'].loc[data['TotalCharges'] == " "] = '0.0'
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    #one hot encoding
    ce_one = ce.OneHotEncoder(cols=category_columns) 
    train_encoded = ce_one.fit_transform(data)

    with open('encoding.pkl', 'wb') as f:
        pickle.dump(ce_one, f)
    
    return data_encoded

    
def train(self, data_encoded):
    #split the data into x and y
    # remove 2nd column for binary y variable and split out into X and y
    X = data_encoded.drop(['Churn_1','Churn_2'], axis=1)
    y = data_encoded['Churn_1']

    #train_test_split
    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    #initialize and fit the decisiontreeclassifier
    dtc = tree.DecisionTreeClassifier(max_depth=5,random_state=42,criterion='gini')
    with open('model.pkl', 'wb') as f:
        pickle.dump(dtc, f)

    dtc.fit(X_train,y_train)
    return dtc


def predict(self, dtc, X_test, y_test):
    # Make predictions here
    y_pred = dtc.predict(X_test)
    mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    #print('DecisionTreeClassifier accuracy score: {}'.format(accuracy))
    return accuracy, y_pred

Encoding(data)
train(data_encoded)
predict(dtc, X_test, y_test)

