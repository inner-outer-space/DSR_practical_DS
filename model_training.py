import pandas as pd
import pickle as pickle

import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# get training and validation data
train = pd.read_csv("./data/training_data.csv")
val = pd.read_csv("./data/validation_data.csv")

# define categorical columns
category_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod', 'Churn']

####### ORIGINAL TRAINING ENCODING #######
 
def encode_training(train_data, category_columns=category_columns):
    # drop customer ID: not a feature for training
    train_data.drop("customerID", axis=1, inplace=True)

    # deal witht the space in TotalCharges
    train_data['TotalCharges'].loc[train_data['TotalCharges'] == " "] = '0.0'
    train_data['TotalCharges'] = pd.to_numeric(train_data['TotalCharges'], errors='coerce')
    
    # create a OneHotEncoder object
    column_mapper = {}  
    encoded_columns = []
    for col in category_columns:
        ohe = OneHotEncoder(drop='if_binary')
        encoded_column = ohe.fit_transform(train_data[[col]])
        column_mapper.update({col: ohe})
        encoded_columns.append(pd.DataFrame(encoded_column.toarray(), columns=ohe.get_feature_names_out([col])))

    # combine the encoded columns with the original dataframe
    encoded_df = pd.concat([train_data.drop(category_columns, axis=1)] + encoded_columns, axis=1)

    return encoded_df, column_mapper

train_encoded, column_mapper = encode_training(train, category_columns=category_columns)

with open('encoding.pkl', 'wb') as f:
    pickle.dump(column_mapper, f)


####### LATER DATA PREPROCESSING BASED ON TRAINING ENCODING#######
def preprocessing(data, encoder_dict):
    # drop customer ID: not a feature for training
    data.drop("customerID", axis=1, inplace=True)
 
    # deal witht the space in TotalCharges
    data['TotalCharges'].loc[data['TotalCharges'] == " "] = '0.0'
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.to_csv('my_data.csv')
    
    # apply the column mapper
    for col in category_columns:
        ohe = encoder_dict[col]
        encoded_column = ohe.transform(data[[col]])
        encoded_df = pd.DataFrame(encoded_column.toarray(), columns=ohe.get_feature_names_out([col]))
        data = pd.concat([data.drop(col, axis=1), encoded_df], axis=1)

    return data

val_encoded = preprocessing(val, column_mapper)

####### MODEL TRAINING  #######    
def train(train_encoded, val_encoded):
    #split the data into x and y for training and testing
    X_train = train_encoded.drop(['Churn_Yes'], axis=1)
    y_train = train_encoded['Churn_Yes']
    X_test = val_encoded.drop(['Churn_Yes'], axis=1)
    y_test = val_encoded['Churn_Yes']
    
    #initialize and fit the decisiontreeclassifier
    dtc = tree.DecisionTreeClassifier(max_depth=5,random_state=42,criterion='gini')
    dtc.fit(X_train,y_train)

    y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    with open('output.txt', 'w') as f:
    # Redirect print statements to the file
        f.write(X_train.to_string(index=False, col_space=10))

    return dtc, accuracy


dtc, accuracy = train(train_encoded, val_encoded)
print('DecisionTreeClassifier accuracy score: {}'.format(accuracy))


# save the model to pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(dtc, f)

def predict(dtc, test_data):
    # Make predictions here
    y_pred = dtc.predict(test_data)
    return  y_pred

