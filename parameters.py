#This file is use for storing parameters
import os
from datetime import datetime

hop_length = 512
target_sr = 11025
script_dir = os.path.dirname(os.path.abspath(__file__))
encoder_path = f'{script_dir}/model/y_Encoder.pkl'

def get_time()->str:
    #Get the current date and time
    now = datetime.now()
    #Format the time as a string
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time