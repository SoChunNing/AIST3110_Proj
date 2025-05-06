# Automatic Chord Detection
Due to the uploading size limit from GitHub, the dataset will be provided via Google Drive.</br>
[Click here](https://drive.google.com/file/d/1uQ6n_Xx5h4Hlnx7VFeIJW57lDM2SVjlE/view?usp=sharing)</br>
After downloading the dataset, you are advised to run the preprocess.py to preprocess the training data first</br>
```
python preprocess.py
```
(Optional, as it will take plenty of time) Run this command to train the RF model:
```
python rf_train.py
```
and the RNN model as well:
```
python rnn_train.py
```
The trained model can be found at ./model</br>
To run the chord_detector, run:
```
python chord_detection.py
```
and follow the guidelines shows on the screen
