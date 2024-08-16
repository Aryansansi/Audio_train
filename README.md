# Emotion Recognition from Audio Data

This project aims to recognize emotions from audio data using machine learning techniques. The project includes multiple Python scripts to process audio files, extract features, train a model, and make predictions.

## Project Structure

```
Emotion_Recognition_Project/
│
├── Processed_Data/
│   ├── extracted_features.csv
│   ├── emotion_recognition_model.joblib
│   ├── processed_audio_data.csv
│   └── emotion_predictions.csv
│
├── Audio_data_processing.py
├── Audio_training.py
├── model_utils.py
└── predict_emotions.py
```

## Files Description

### 1. `Audio_data_processing.py`

This script processes raw audio files to extract basic information such as sample rate, audio length, and amplitude. It also visualizes different aspects of the audio data, including the waveform, trimmed waveform, spectrogram, and mel spectrogram. The processed data is saved in a CSV file (`processed_audio_data.csv`).

#### Key Functions:
- **Audio Loading:** The script loads each `.wav` file using `librosa`.
- **Data Collection:** Collects information like actor ID, file name, sample rate, and audio amplitude.
- **Visualization:** Displays various visualizations for the first audio file processed.

### 2. `Audio_training.py`

This script extracts features from the audio files, such as MFCCs, Chroma features, Spectral Contrast, and Zero-Crossing Rate, and trains a RandomForestClassifier model. The model is then saved for later use.

#### Key Functions:
- **Feature Extraction:** Extracts and computes various audio features using `librosa`.
- **Model Training:** Trains a RandomForestClassifier on the extracted features.
- **Data Splitting:** Splits the data into training, validation, and test sets.
- **Model Evaluation:** Evaluates the model on validation and test sets and saves the model in a joblib file (`emotion_recognition_model.joblib`).

### 3. `model_utils.py`

This script contains utility functions for loading the model, extracting features, and predicting emotions from new audio files. It also includes a function to plot and save a confusion matrix of the model's predictions.

#### Key Functions:
- **Model Loading:** Loads the trained model.
- **Feature Extraction:** Extracts features for prediction.
- **Emotion Prediction:** Predicts emotions for a given audio file.
- **Confusion Matrix Plotting:** Plots and saves the confusion matrix to a PNG file.

### 4. `predict_emotions.py`

This script uses the trained model to predict emotions from new audio files. It processes the first five audio files from each actor, predicts the emotion, and saves the predictions to a CSV file (`emotion_predictions.csv`).

#### Key Functions:
- **Emotion Prediction:** Iterates over each actor's audio files, predicts emotions, and saves the results.
- **Result Saving:** Stores the predictions along with confidence scores in a CSV file.

## Installation

To run the project, you need to install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn librosa scikit-learn joblib
```

## How to Use

1. **Process Audio Data:**
   - Run `Audio_data_processing.py` to process the audio files and visualize the data.
   - The processed data will be saved in `Processed_Data/processed_audio_data.csv`.

2. **Train the Model:**
   - Run `Audio_training.py` to extract features and train the model.
   - The model will be saved as `Processed_Data/emotion_recognition_model.joblib`.

3. **Predict Emotions:**
   - Use `predict_emotions.py` to predict emotions from new audio files.
   - Predictions will be saved in `Processed_Data/emotion_predictions.csv`.

4. **Utilities:**
   - Use `model_utils.py` for additional utilities like confusion matrix plotting or model loading.

## License

This project is licensed under the MIT License.

---

This README provides an overview of your project, its structure, and instructions for running the code. Let me know if you need any changes!
