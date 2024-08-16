import os
import pandas as pd
import numpy as np
import librosa
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Define the base path to your audio dataset
base_dataset_path = 'Audio_dataset'
features_output_file = 'Processed_Data/extracted_features.csv'
model_output_file = 'Processed_Data/emotion_recognition_model.joblib'

# Function to extract features from an audio file
def extract_features(y, sr):
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

    # Extract Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # Combine all features
    features = np.concatenate((mfccs_mean, chroma_mean, spectral_contrast_mean, [zcr_mean]))
    
    return features

# Initialize lists to hold features and labels
X = []
y = []

# Check if features file exists and is not empty
if os.path.exists(features_output_file) and os.path.getsize(features_output_file) > 0:
    print(f"Features file {features_output_file} already exists. Skipping feature extraction.")
else:
    # Iterate over each actor folder (Actor_01 to Actor_24)
    for actor_id in range(1, 25):
        actor_folder = f'Actor_{actor_id:02d}'
        dataset_path = os.path.join(base_dataset_path, actor_folder)

        # Use glob to find all audio files in the current actor's folder
        audio_files = glob(os.path.join(dataset_path, '*.wav'))

        # Process each audio file in the actor's folder
        for audio_file in audio_files:
            # Load the audio file using librosa
            y_audio, sr = librosa.load(audio_file)

            # Extract features from the audio
            features = extract_features(y_audio, sr)
            X.append(features)

            # Label the data with the actor ID
            y.append(actor_id)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Convert the feature matrix and labels to a DataFrame
    features_df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
    features_df['Label'] = y

    # Save the extracted features and labels to a CSV file
    features_df.to_csv(features_output_file, index=False)
    print(f"Extracted features and labels saved to {features_output_file}")

# Load the extracted features and labels
features_df = pd.read_csv(features_output_file)
X = features_df.drop('Label', axis=1).values
y = features_df['Label'].values

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize and train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_pred = clf.predict(X_val)
print("Validation Set Performance:")
print(classification_report(y_val, y_val_pred))

# Evaluate the model on the test set
y_test_pred = clf.predict(X_test)
print("Test Set Performance:")
print(confusion_matrix(y_test, y_test_pred))
print(f'Accuracy: {accuracy_score(y_test, y_test_pred)}')

# Save the trained model
joblib.dump(clf, model_output_file)
print(f"Model saved to {model_output_file}")
