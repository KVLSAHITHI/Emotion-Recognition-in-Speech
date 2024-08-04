# Emotion-Recognition-in-Speech


1. GUI Code (gui.py)
This script creates a graphical user interface (GUI) for the speech emotion recognition application.

Functionality:

The create_gui function sets up a Tkinter window with a button to upload an audio file.
When the button is clicked, it opens a file dialog to select an audio file and calls the browse_file function.
The browse_file function takes the selected file, uses the predict_emotion_func to predict the emotion, and displays the result in a message box.
How to Execute:

Ensure you have tkinter installed.
Save the script as gui.py.
Import the predict_emotion_func from the main script (described below).
Run the script: python gui.py


2. Training and Testing Code (train_test.py)
This script trains a speech emotion recognition model using Support Vector Machine (SVM).

Functionality:

Loads a CSV file containing metadata about the audio files.
Extracts Mel-Frequency Cepstral Coefficients (MFCC) features from the audio files.
Trains an SVM classifier on the extracted features and corresponding labels.
Saves the trained model and label encoder using pickle.
How to Execute:

Ensure you have the required libraries: pandas, numpy, librosa, scikit-learn, pickle.
Save the script as train_test.py.
Update the paths to your CSV file and audio files directory.
Run the script: python train_test.py.
This script will generate two files: model.pkl (trained model) and label_encoder.pkl (label encoder).


3. Main Script (main.py)
This script is the main application that uses the trained model to predict emotions from audio files.

Functionality:

Loads the trained model and label encoder.
Defines functions to extract MFCC features and predict the emotion from an audio file.
Creates a Tkinter GUI to upload an audio file and display the predicted emotion.
How to Execute:

Ensure you have tkinter, librosa, and pickle installed.
Save the script as main.py.
Make sure model.pkl and label_encoder.pkl (generated from train_test.py) are in the same directory as main.py.
Run the script: python main.py
