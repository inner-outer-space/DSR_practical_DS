import pickle as pickle
import pandas as pd
import model_training as mt


def generate_predictions(data):
    # encode the data 
    col_mapper = pickle.load(open('encoding.pkl','rb'))

    #data_encoded = encode(data)
    data_encoded = mt.preprocessing(data, col_mapper)
    data_encoded = data_encoded.drop(['Churn_Yes'], axis=1)
    data_encoded = data_encoded.reset_index(drop=True, inplace=True)
    print(data_encoded.head())

    # Loading model to compare the results
    model = pickle.load(open('model.pkl','rb'))
    prediction = model.predict(data_encoded)
    #prediction = make_predictions(data_encoded, model)
    
    return data_encoded #prediction



