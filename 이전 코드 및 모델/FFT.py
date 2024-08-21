import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from pydub import AudioSegment
import os
import matplotlib as mpl

# 한글 지원 글꼴로 변경
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# Function to convert any audio file format to a numpy array
def read_audio(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    # 지원하지 않는 파일 형식은 무시
    if file_extension not in ['.mp3', '.wav', '.flv', '.mp4', '.ogg', '.flac', '.m4a']:
        return None, None

    audio = AudioSegment.from_file(file_path)
    data = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        if len(data) % 2 != 0:  # 샘플 수가 홀수인 경우 마지막 샘플 제거
            data = data[:-1]
        data = data.reshape((-1, 2))
        data = data.mean(axis=1)  # 스테레오 채널을 평균내어 모노 데이터로 변환
    return audio.frame_rate, data

# Directory containing the audio files
directory_path = 'C:/Users/SKT038/Desktop/유튜브/말벌'

# Process each audio file in the directory
for filename in os.listdir(directory_path):
    audio_file_path = os.path.join(directory_path, filename)

    # Read the audio file
    fs, data = read_audio(audio_file_path)
    
    # If audio data is None, skip the file
    if data is None:
        continue

    # Perform FFT and create frequency axis
    fft_values = fft(data)
    fft_freqs = fftfreq(len(data), 1/fs)
    fft_magnitude = np.abs(fft_values)

    # Filter frequencies (0 Hz - 3000 Hz)
    freq_mask = (fft_freqs >= 0) & (fft_freqs <= 3000)
    filtered_freqs = fft_freqs[freq_mask]
    filtered_magnitude = fft_magnitude[freq_mask]

    # Visualize FFT results
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_freqs, filtered_magnitude)
    plt.title(f'FFT Magnitude (0 - 3000 Hz) for {filename}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xticks(np.arange(0, 3100, 100))
    plt.show()

    # Print the file name after processing
    print(f"Processed audio file: {audio_file_path}")

print("All FFT plots have been processed.")
