# Automatic Chord Recognition

This project implements an Automatic Chord Recognition (ACR) system using machine learning and deep learning techniques. It provides tools for preprocessing audio data, training models using Random Forest and LSTM (Long Short-Term Memory) networks, and evaluating the performance of the trained models.

## Project Overview

The goal of this project is to identify musical chords from audio recordings. The system processes audio files to extract features (likely chromagrams), trains classifiers to predict chord labels, and evaluates the results against ground truth annotations.

### Key Features
- **Audio Preprocessing**: Utilities to load audio, remove vocals (harmonic-percussive source separation), and simplify chord labels.
- **Machine Learning Models**:
  - **Random Forest**: A baseline classifier implemented using `scikit-learn`.
  - **RNN (LSTM)**: A deep learning model implemented using `PyTorch` for sequence modeling.
- **Evaluation**: Calculation of the Chord Sequence Recognition (CSR) score to measure model performance.
- **Data Handling**: Custom dataset and dataloader implementations for handling variable-length audio sequences.

## Project Structure

```
chord_recognition/
├── chord_detection.py    # Main script for performing chord detection and calculating CSR scores.
├── parameters.py         # Configuration file containing hyperparameters and file paths.
├── preprocess.py         # Functions for audio loading, vocal removal, and label conversion.
├── rf_train.py           # Script to train and evaluate the Random Forest model.
├── rnn_model.py          # Definition of the LSTMClassifier PyTorch model.
├── rnn_train.py          # Script to train the RNN/LSTM model.
├── y_encoder.py          # Utilities for encoding and decoding chord labels.
├── requirements.txt      # List of Python dependencies.
├── audio_to_regconize/   # Directory for input audio and result files.
└── model/                # Directory to save trained models (e.g., lstm_model.pth, RF_model.pkl).
```

## Installation

1. Clone the repository.
2. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocessing & Configuration
- Adjust hyperparameters (sample rate, hop length, model parameters) in `parameters.py`.
- Ensure your audio data and ground truth files are placed in the appropriate directories as expected by the scripts (check `parameters.py` and `preprocess.py` for path details).
- **Note**: Due to the uploading size limit from GitHub, the dataset will be provided via Google Drive. [Click here](https://drive.google.com/file/d/1uQ6n_Xx5h4Hlnx7VFeIJW57lDM2SVjlE/view?usp=sharing) to download.

### 2. Training the Random Forest Model
To train the Random Forest baseline:

```bash
python rf_train.py
```
This script will load CSV data, train the model, print the accuracy, and save the model to `model/RF_model.pkl`.

### 3. Training the LSTM Model
To train the deep learning model:

```bash
python rnn_train.py
```
This script handles data loading, training the LSTM network, and saving the model checkpoints.

### 4. Chord Detection & Evaluation
To run the chord detection pipeline and calculate CSR scores:

```bash
python chord_detection.py
```

## Dependencies

- **librosa**: For audio analysis and feature extraction.
- **numpy & pandas**: For data manipulation.
- **scikit-learn**: For the Random Forest classifier and label encoding.
- **torch**: For the LSTM model implementation.
- **joblib**: For saving/loading scikit-learn models and encoders.
- **tqdm**: For progress bars during training.
- **matplotlib**: For plotting (used in training scripts).

## Notes
- The system supports 13 chord classes (12 major/minor roots + 'N' for no chord).
- Vocal removal is applied during preprocessing to improve chord extraction accuracy.
