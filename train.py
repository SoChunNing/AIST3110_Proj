#This file is used for model training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
import joblib
from parameters import *
from datetime import datetime


def load_csv(script_dir)->pd.DataFrame:
    #Load data from all csv files

    #Create an empty DataFrame first
    data_load = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'y'])
    for dataset in os.listdir(f'{script_dir}/data/csv'):
        for track in os.listdir(f'{script_dir}/data/csv/{dataset}'):
            for song in os.listdir(f'{script_dir}/data/csv/{dataset}/{track}'):
                data = pd.read_csv(f'{script_dir}/data/csv/{dataset}/{track}/{song}')
                data_load = pd.concat([data_load, data], ignore_index=True)
                print('loading ' f'{script_dir}/data/csv/{dataset}/{track}/{song}')

    return data_load

def get_time()->str:
    #Get the current date and time
    now = datetime.now()
    #Format the time as a string
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time

if __name__ == "__main__":
    data = load_csv(script_dir)
    y = data['y']
    data = data.drop(['y'], axis=1)
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(data, y_encoded, test_size=0.2, random_state=42)
    print("Start training model at", get_time())
    model = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42, max_features='log2')
    model.fit(X_train, y_train)
    
    #Evaluate the model
    y_pred = model.predict(X_test)
    accuarcy = accuracy_score(y_test, y_pred)
    print(f'Training complete at {get_time()}, model accuracy = {accuarcy:.2f}')

    #Save the model and y_encoder
    joblib.dump(model, f'{script_dir}/RF_model.pkl')
    joblib.dump(label_encoder, f'{script_dir}/y_Encoder.pkl')

