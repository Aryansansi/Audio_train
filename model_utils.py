import joblib
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Define the path to your saved model and other files
model_filename = 'Processed_Data/emotion_recognition_model.joblib'
features_output_file = 'Processed_Data/extracted_features.csv'
confusion_matrix_output_file = 'Processed_Data/confusion_matrix.png'

# Function to load the model
def load_model():
    return joblib.load(model_filename)

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

# Function to predict emotion from an audio file
def predict_emotion(audio_file):
    model = load_model()
    y, sr = librosa.load(audio_file)
    features = extract_features(y, sr)
    features = np.reshape(features, (1, -1))  # Reshape for prediction
    prediction = model.predict(features)
    return prediction[0]

# Function to plot and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, output_file=confusion_matrix_output_file):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 25), yticklabels=range(1, 25))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(output_file)
    plt.show()

# Example usage of the prediction function
if __name__ == '__main__':
    # Load the features and labels
    features_df = pd.read_csv(features_output_file)
    X = features_df.drop('Label', axis=1).values
    y = features_df['Label'].values

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Load the model and predict
    model = load_model()
    y_test_pred = model.predict(X_test)

    # Plot and save the confusion matrix
    plot_confusion_matrix(y_test, y_test_pred)
