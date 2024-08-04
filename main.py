import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
with open('model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Function to extract MFCC features from an audio file
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

# Function to predict emotion
def predict_emotion(audio_file):
    features = extract_features(audio_file)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = classifier.predict(features)
        emotion = label_encoder.inverse_transform(prediction)
        return emotion[0]
    else:
        return "Error in feature extraction."

# GUI setup
def browse_file():
    filename = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if filename:
        emotion = predict_emotion(filename)
        messagebox.showinfo("Predicted Emotion", f"The emotion in the speech is: {emotion}")

# Create the main window
root = tk.Tk()
root.title("Speech Emotion Recognition")

# Add a button to browse and predict
browse_button = tk.Button(root, text="Upload Audio File", command=browse_file)
browse_button.pack(pady=20)

# Run the application
root.mainloop()
