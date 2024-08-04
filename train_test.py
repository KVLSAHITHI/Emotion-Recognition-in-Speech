import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Load CSV file
csv_file_path = r"C:\Users\srivi\OneDrive\Desktop\nlp\speech_emotions.csv"
audio_files_dir = r"C:\Users\srivi\OneDrive\Desktop\nlp\files"

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Print the first few rows of the dataframe to verify content
print("CSV File Content:")
print(df.head())

# Function to extract MFCC features
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

# Extract features from audio files
features = []
labels = []

# Iterate through all subdirectories in the audio files directory
for folder_name in os.listdir(audio_files_dir):
    folder_path = os.path.join(audio_files_dir, folder_name)

    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            audio_file_path = os.path.join(folder_path, file_name)

            # Check if the file exists and is an audio file
            if os.path.isfile(audio_file_path):
                # Use only the file name without extension for the label
                label = file_name.split('.')[0]  # Assumes file names have no extensions
                if file_name.endswith('.wav'):  # Ensure the file has a .wav extension
                    feature = extract_features(audio_file_path)
                    if feature is not None:
                        features.append(feature)
                        labels.append(label)
                else:
                    print(f"Skipping non-wav file: {audio_file_path}")

# Convert lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# Print the number of collected features and labels
print(f"Number of features collected: {len(features)}")
print(f"Number of labels collected: {len(labels)}")

# Check if data is empty
if X.shape[0] == 0 or y.shape[0] == 0:
    print("No data to split. Please check feature extraction and label collection.")
else:
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM classifier
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)

    # Save the trained model and label encoder
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(classifier, model_file)

    with open('label_encoder.pkl', 'wb') as le_file:
        pickle.dump(le, le_file)

    # Print accuracy on test data
    accuracy = classifier.score(X_test, y_test)
    print(f"Model accuracy on test data: {accuracy * 100:.2f}%")
