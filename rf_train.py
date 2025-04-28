#This file is used for model training using Random Forest Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
import joblib
from parameters import *
from y_encoder import *

def load_csv()->list:
    #Load data from all csv files
    #Return a list of dataframes
    data_list = []
    for dataset in os.listdir(f'{script_dir}/data/csv'):
        for track in os.listdir(f'{script_dir}/data/csv/{dataset}'):
            for song in os.listdir(f'{script_dir}/data/csv/{dataset}/{track}'):
                data = pd.read_csv(f'{script_dir}/data/csv/{dataset}/{track}/{song}')
                data_list.append(data)
                print('loading 'f'{script_dir}/data/csv/{dataset}/{track}/{song}')

    return data_list

if __name__ == "__main__":
    data_list = load_csv()
    data = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'y'])
    for df in data_list:
        data = pd.concat([data, df], ignore_index=True)
    y = data['y']
    data = data.drop(['y'], axis=1)
    create_encoder(y)
    #Encode the labels using the fitted encoder
    y_encoded = encode_y(y)
    print(set(y))
    print(len(set(y_encoded)))
    print("Start training model at", get_time())
    #Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, y_encoded, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_features='log2')
    model.fit(X_train, y_train)
    
    #Evaluate the model
    y_pred = model.predict(X_test)
    accuarcy = accuracy_score(y_test, y_pred)
    print(f'Training complete at {get_time()}, model accuracy = {accuarcy:.2f}')

    #Save the model
    joblib.dump(model, f'{script_dir}/model/RF_model.pkl')

