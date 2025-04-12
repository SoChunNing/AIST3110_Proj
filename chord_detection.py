#This is the main file for my automatic chord detector
from preprocess import load_audio
from parameters import *
from sklearn.preprocessing import LabelEncoder
from preprocess import load_gt
import numpy as np
import joblib
import os
from y_encoder import decode_y
from rnn_model import LSTMClassifier
import torch

def calculate_csr(gt_path, result_path):
    # Calculate the CSR (Chord Sequence Recognition) score
    result_duration = 0.0
    gt_start_time_list, gt_end_time_list, gt_chord_list = load_gt(gt_path)
    result_start_time_list, result_end_time_list, result_chord_list = load_gt(result_path)
    #CSR: total duration of segments where annotation equals estimation / total duration of annotated segements
    for i in range(len(gt_chord_list)):
        start_time, end_time, chord = float(gt_start_time_list[i]), float(gt_end_time_list[i]), gt_chord_list[i]
        # Find the corresponding segment in the result
        for j in range(len(result_chord_list)):
            result_start_time, result_end_time, result_chord = float(result_start_time_list[j]), float(result_end_time_list[j]), result_chord_list[j]
            # Check if the chords match and the time intervals overlap
            if chord == result_chord and not (end_time < result_start_time or start_time > result_end_time):
                # Calculate the overlap duration
                overlap_duration = min(end_time, result_end_time) - max(start_time, result_start_time)
                break
        else:
            overlap_duration = 0.0
        # Add the overlap duration to the CSR score
        result_duration += overlap_duration
    # Calculate the total duration of the annotated segments
    totol_duration = gt_end_time_list[-1]
    csr_score = result_duration / totol_duration if totol_duration > 0 else 0.0 # Avoid division by zero
    print(f"CSR score: {csr_score:.2f}")

def find_gt(song_name):
    found = False
    # Scan through the datasets to see if the song exists in the ground truth file
    for dataset in os.listdir(f'{script_dir}/data/gt'):
        for track in os.listdir(f'{script_dir}/data/gt/{dataset}'):
            for song in os.listdir(f'{script_dir}/data/gt/{dataset}/{track}'):
                gt_name = os.path.splitext(song)[0]
                if gt_name == song_name:
                    gt_path = f'{script_dir}/data/gt/{dataset}/{track}/{song}'
                    calculate_csr(gt_path, f'{script_dir}/audio_to_regconize/result/{song_name}.lab')
                    found = True
                    break
            if found:
                break
        if found:
            break
    if not found:
        print(f"Ground truth for {song_name} not found in dataset.")

def save_chord_to_lab(chord_result, song_name):
    os.makedirs(f'{script_dir}/audio_to_regconize/result', exist_ok=True)
    np.savetxt(f'{script_dir}/audio_to_regconize/result/{song_name}.lab', chord_result, delimiter=",", fmt='%s')
    print('Result save at:', f'{script_dir}/audio_to_regconize/result/{song_name}.lab')

def extarct_chord_rf(audio_path, song_name):
    # Extract chord using Random Forest model
    X =  load_audio(audio_path, hop_length, target_sr)
    model = joblib.load(f'{script_dir}/model/rf_model.pkl')
    y = model.predict(X)
    y_decode = decode_y(y)
    chord_results = []
    start_time = 0.0
    tw = (hop_length / target_sr)

    prev_chord = y_decode[0]
    print(f"Chord results for {song_name}.mp3:")
    for i, chord in enumerate(y_decode, 1):
        #Skip the current loop if the chord label doesn't change!
        if chord == prev_chord and i != len(y_decode):
            continue
        end_time = i * tw
        print(f'{start_time:.6f}  {end_time:.6f}  {prev_chord}')
        chord_results.append(f'{start_time}	 {end_time}	 {prev_chord}')
        start_time = end_time
        prev_chord = chord

    save_chord_to_lab(chord_results, song_name)

def extarct_chord_rnn(audio_path, song_name):
    # Extract chord using RNN model
    X = load_audio(audio_path, hop_length, target_sr)
    model = LSTMClassifier(input_size, hidden_dim, num_classes, num_layers, use_cuda=True, bidirectional=True)
    model.load_state_dict(torch.load(f'{script_dir}/model/lstm_model.pth'))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        X = torch.tensor(X, dtype=torch.float64).cuda()
    else:
        X = torch.tensor(X, dtype=torch.float64)
    output = model(X)
    pred = output.topk(1, dim=2)[1].squeeze().view(-1)
    y = pred.tolist()
    y_decode = decode_y(y)
    chord_results = []
    start_time = 0.0
    tw = (hop_length / target_sr)

    prev_chord = y_decode[0]
    print(f"Chord results for {song_name}.mp3:")
    for i, chord in enumerate(y_decode, 1):
        #Skip the current loop if the chord label doesn't change!
        if chord == prev_chord and i != len(y_decode):
            continue
        end_time = i * tw
        print(f'{start_time:.6f}  {end_time:.6f}  {prev_chord}')
        chord_results.append(f'{start_time}	 {end_time}	 {prev_chord}')
        start_time = end_time
        prev_chord = chord

    save_chord_to_lab(chord_results, song_name)

if __name__ == "__main__":
    print("Welcome to the automatic chord detector!")
    print("Reminder: Please put the audio file under '/audio_to_regconize/audio/'")
    model_id = input("Enter the model ID(1 for RF, 2 for RNN): ")
    cal_csr = input("Do you want to calculate the CSR? (Y/N): ").strip().upper()
    if model_id != '1' and model_id != '2':
        print("Invalid model ID. Please enter 1 or 2.")
        exit(1)
    #The program will scan through '/audio_to_regconize/audio/' and detect all the audio files
    for song in os.listdir(f'{script_dir}/audio_to_regconize/audio'):
        song_name= os.path.splitext(song)[0]
        if model_id == '1':
            extarct_chord_rf(f'{script_dir}/audio_to_regconize/audio/{song}', song_name)
        elif model_id == '2':
            extarct_chord_rnn(f'{script_dir}/audio_to_regconize/audio/{song}', song_name)
        if cal_csr == 'Y':
            find_gt(song_name)
