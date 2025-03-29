#This file is use to process the audio/gt file into training data
import librosa
import numpy as np
import pandas as pd
import os

#Extract start time, endtime and chord label from ground-truth file
def load_gt(gt_path):

    start_time_list, end_time_list, chord_list = [], [], []

    with open(gt_path, 'r') as f:
        for gt_chord_line in f:
            gt_chord_line = gt_chord_line.rstrip() #remove all the whitespace at the right
            str_toks = gt_chord_line.split()
            start_time = float(str_toks[0])
            start_time_list.append(start_time)
            end_time = float(str_toks[1])
            end_time_list.append(end_time)
            chord = str_toks[2]
            chord_list.append(chord)
    
    return start_time_list, end_time_list, chord_list
    
#Extract chromagram of an audio file
def load_audio(audio_path, hop_length, target_sr = 44100):

    y, sr = librosa.load(audio_path, sr = None)
    y_downsampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    tuning = librosa.estimate_tuning(y=y_downsampled, sr=target_sr)
    y_downsampled_pitched = librosa.effects.pitch_shift(y=y_downsampled, sr=target_sr, n_steps=0) #pitch for debugging
    chromagram = librosa.feature.chroma_cqt(y=y_downsampled_pitched, sr=target_sr, hop_length=hop_length, tuning=tuning)
    chromagram_T = chromagram.T #row: time frame, column: pitch-class
    
    return chromagram_T

#Label y(chord) to chromagram within the coreesponding boundaries and return a pd.DataFrame
def label_y_to_chromagram(start_time_list, end_time_list, chord_list, chromagram:np.array, hop_length, sr)->pd.DataFrame:
    #time length of each chroma frame
    y = []
    tw = hop_length/sr
    chromagram_length = chromagram.shape[0] 
    for i in range(len(start_time_list)):
        #both the lists of start time, end time and chord have the same length
        start_time, end_time, chord = start_time_list[i], end_time_list[i], chord_list[i]
        start_bound = round(start_time / tw)
        end_bound = round(end_time / tw)
        y.extend((end_bound - start_bound) * [chord])
        #check if the length of y is equal to the length of chromagram (no. of rows)
    if len(y) < chromagram_length:
        y.extend((chromagram_length - len(y)) * 'N') #Fit with No chord first, will be adjust later
    else:
        #Cut the redundant part
        y = y[:chromagram_length]
            
    df = pd.DataFrame(chromagram)
    df['y'] = y
    return df
    

if __name__ == "__main__":
    ###Load the data
    new_sr = 11025
    hop_length = 512

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for dataset in os.listdir(f'{script_dir}/data/audio'):
        for track in os.listdir(f'{script_dir}/data/audio/{dataset}'):
            #Create the directory for csv files
            os.makedirs(f'{script_dir}/data/chromagram/{dataset}/{track}', exist_ok=True) 
            for song in os.listdir(f'{script_dir}/data/audio/{dataset}/{track}'):
                chromagram = load_audio(f'{script_dir}/data/audio/{dataset}/{track}/{song}', hop_length, new_sr)
                song_name= os.path.splitext(song)[0]
                start_time_list, end_time_list, chord_list = load_gt(f'{script_dir}/data/gt/{dataset}/{track}/{song_name}.lab')
                data = label_y_to_chromagram(start_time_list, end_time_list, chord_list, chromagram, hop_length, new_sr)
                write_path = f'{script_dir}/data/chromagram/{dataset}/{track}/{song_name}.csv'
                data.to_csv(write_path, index=False)
                print(f'{script_dir}/data/audio/{dataset}/{track}/{song} conversion done!')

    print('All Done!!!')