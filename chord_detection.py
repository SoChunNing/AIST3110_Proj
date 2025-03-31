#This is the main file for my automatic chord detector
from preprocess import load_audio
from parameters import *
from sklearn.preprocessing import LabelEncoder
import joblib

def extarct_chord(audio_path, model):

    X =  load_audio(audio_path, hop_length, target_sr)
    y = model.predict(X)
    encoder = joblib.load(f'{script_dir}/label_encoder.pkl')

    # Use inverse_transform() to decode y
    y_decode = encoder.inverse_transform(y)

    chord_results = []
    start_time = 0.0
    tw = (hop_length / target_sr)

    prev_chord = y_decode[0]
    for i, chord in enumerate(y_decode, i):
        #Skip the current loop if the chord label doesn't change!
        if chord == prev_chord and i != len(y_decode):
            continue

        end_time = i * tw
        chord_results.append(f'{start_time}	{end_time}	{prev_chord}')
        start_time = end_time
        prev_chord = chord

    print(chord_results)

def save_chord_to_lab(save_path):
    pass