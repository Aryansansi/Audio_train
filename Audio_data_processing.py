import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import librosa
import librosa.display
import os

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Define the base path to your audio dataset
base_dataset_path = 'Audio_dataset'
output_file = 'Processed_Data/processed_audio_data.csv'

# Initialize a list to collect all data
all_data = []

# Variable to control whether to display the graph
display_first_graph = True

# Iterate over each actor folder (Actor_01 to Actor_24)
for actor_id in range(1, 25):
    actor_folder = f'Actor_{actor_id:02d}'
    dataset_path = os.path.join(base_dataset_path, actor_folder)

    # Use glob to find all audio files in the current actor's folder
    audio_files = glob(os.path.join(dataset_path, '*.wav'))

    # Process each audio file in the actor's folder
    for i, audio_file in enumerate(audio_files):
        # Load the audio file using librosa
        y, sr = librosa.load(audio_file)
        file_name = os.path.basename(audio_file).split('.')[0]

        # Prepare a dictionary for the current file's data
        data_dict = {
            'Actor': actor_folder,
            'File': file_name,
            'Sample_Rate': sr,
            'Audio_Length': len(y),
            'Amplitude': y,
        }

        # Append the raw audio data as a DataFrame to the list
        audio_df = pd.DataFrame(data_dict)
        all_data.append(audio_df)

        # Display graph only for the first file of the first actor
        if display_first_graph:
            # Plotting raw audio data
            pd.Series(y).plot(figsize=(10, 5), lw=1, title=f"Raw Audio Example: {actor_folder}", color=color_pal[0])
            plt.show()

            # Plotting trimmed raw audio data
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            pd.Series(y_trimmed).plot(figsize=(10, 5), lw=1, title=f"Raw Audio Trimmed Example: {actor_folder}", color=color_pal[1])
            plt.show()

            # Plotting zoomed raw audio data
            pd.Series(y[30000:30500]).plot(figsize=(10, 5), lw=1, title=f"Raw Audio Zoomed Example: {actor_folder}", color=color_pal[2])
            plt.show()

            # Spectrogram
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            # Plot the transformed audio data
            fig, ax = plt.subplots(figsize=(10, 5))
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
            ax.set_title(f'Spectrogram Example: {actor_folder}', fontsize=20)
            fig.colorbar(img, ax=ax, format=f'%0.2f')
            plt.show()

            # Mel Spectrogram
            S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128 * 2)
            S_db_mel = librosa.amplitude_to_db(S_mel, ref=np.max)

            # Plot mel spectrogram
            fig, ax = plt.subplots(figsize=(15, 5))
            img = librosa.display.specshow(S_db_mel, x_axis='time', y_axis='log', ax=ax)
            ax.set_title(f'Mel Spectrogram Example: {actor_folder}', fontsize=20)
            fig.colorbar(img, ax=ax, format=f'%0.2f')
            plt.show()

            # Disable further graph display
            display_first_graph = False

# Concatenate all the data into a single DataFrame
final_df = pd.concat(all_data, ignore_index=True)

# Save the final DataFrame to a CSV file
final_df.to_csv(output_file, index=False)

print(f"Processed data saved to {output_file}")
