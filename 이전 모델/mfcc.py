import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# list containing all the features
extracted = []

# Define the audio dataset path
audio_dataset_path = r"C:\Users\SKT038\Desktop\sound_data\split_sound"

# List the folders (classes) in the dataset path
class_folders = [f for f in os.listdir(audio_dataset_path) if os.path.isdir(os.path.join(audio_dataset_path, f))]

# Process each class folder
for class_name in class_folders:
    class_path = os.path.join(audio_dataset_path, class_name)

    # List the first 10 audio files in each class folder
    audio_files = os.listdir(class_path)[:]

    # Process each audio file
    for audio_file in tqdm(audio_files, desc=f"Processing {class_name}"):
        file_path = os.path.join(class_path, audio_file)

        # Load the audio file
        audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")

        # Extract the features
        feature = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
        break
        # Feature scaling
        scaled_feature = np.mean(feature.T, axis=0)

        # Store the features and class labels in a list
        extracted.append([scaled_feature, class_name])

# Create a new dataframe
#extracted_df = pd.DataFrame(extracted, columns=["feature", "class"])

# Storing the dataframe to pickle for further processing
#extracted_df.to_pickle("extracted_df.pkl")

# Display the first few rows of the dataframe
#extracted_df.head()