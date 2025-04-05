from sklearn.preprocessing import LabelEncoder
from parameters import script_dir
import joblib
import pandas as pd
from parameters import encoder_path

def create_encoder(y: pd.DataFrame) -> LabelEncoder:
    
    encoder = LabelEncoder()
    encoder.fit(y)
    encoder_path = f'{script_dir}/y_Encoder.pkl'
    # Save the encoder to a file
    joblib.dump(encoder, encoder_path)

def encode_y(y):
    # Encode the labels using the fitted encoder
    encoder = joblib.load(encoder_path)
    y_encoded = encoder.transform(y)
    return y_encoded

def decode_y(y_encoded):
    # Decode the labels using the fitted encoder
    encoder = joblib.load(encoder_path)
    y = encoder.inverse_transform(y_encoded)
    return y