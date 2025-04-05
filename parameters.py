#This file is use for storing parameters
import os

hop_length = 512
target_sr = 11025
script_dir = os.path.dirname(os.path.abspath(__file__))
encoder_path = f'{script_dir}/y_Encoder.pkl'