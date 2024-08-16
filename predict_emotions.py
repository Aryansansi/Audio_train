import os
import pandas as pd
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
import joblib
from glob import glob

# Define the base path to your audio dataset
base_dataset_path = 'Audio_dataset'
model_filename = 'Processed_Data/emotion_recognition_model.joblib'

# Emotion mapping based on actor IDs
emotion_map = {
    1: 'Happy',
    2: 'Sad',
    3: 'Angry',
    4: 'Surprised',
    5: 'Neutral',
    6: 'Fearful',
    7: 'Disgusted',
    8: 'Bored',
    9: 'Excited',
    10: 'Frustrated',
    11: 'Content',
    12: 'Annoyed',
    13: 'Proud',
    14: 'Embarrassed',
    15: 'Confident',
    16: 'Nervous',
    17: 'Calm',
    18: 'Irritated',
    19: 'Tired',
    20: 'Euphoric',
    21: 'Disappointed',
    22: 'Ashamed',
    23: 'Hopeful',
    24: 'Resigned'
}

# Load the model
model = joblib.load(model_filename)

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

# Initialize a list to store prediction results
results = []

# Iterate over each actor folder (Actor_01 to Actor_24)
for actor_id in range(1, 25):
    actor_folder = f'Actor_{actor_id:02d}'
    dataset_path = os.path.join(base_dataset_path, actor_folder)

    # Use glob to find all audio files in the current actor's folder
    audio_files = glob(os.path.join(dataset_path, '*.wav'))

    # Process the first 5 audio files from each actor
    for i, audio_file in enumerate(audio_files[:5]):
        # Load the audio file using librosa
        y_audio, sr = librosa.load(audio_file)
        
        # Extract features from the audio
        features = extract_features(y_audio, sr)
        
        # Predict emotion
        prediction_prob = model.predict_proba([features])[0]
        predicted_class = model.predict([features])[0]
        confidence = np.max(prediction_prob)
        emotion = emotion_map.get(predicted_class, 'Unknown')
        
        # Append results
        results.append({
            'File': os.path.basename(audio_file),
            'Predicted Emotion': emotion,
            'Confidence': confidence
        })

# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('Processed_Data/emotion_predictions.csv', index=False)
print(f"Emotion predictions saved to Processed_Data/emotion_predictions.csv")
