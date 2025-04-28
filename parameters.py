#This file is use for storing parameters
import sys
import os
from datetime import datetime

hop_length = 512
target_sr = 11025

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
encoder_path = f'{script_dir}/model/y_Encoder.pkl'

#Hyperparameters
batch_size = 8
num_epochs = 50
input_size = 12
hidden_dim = 128
num_classes = 13 #13 classes of chord labels (including N)
num_layers = 3
lr = 0.1
momentum = 0.8

def get_time()->str:
    #Get the current date and time
    now = datetime.now()
    #Format the time as a string
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time