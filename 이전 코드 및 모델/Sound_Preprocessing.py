import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from pydub import AudioSegment
import os
import matplotlib as mpl
from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import stft

# 오디오 파일 불러오기(fs, data = read_audio(file_path))
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
    return data, audio.frame_rate

# 오디오 파일 분할(split_len 1000당 1초)
def split_audio_file(source_file, save_folder, split_length=2000):
    # 소스 파일에서 오디오를 로드합니다.
    audio = AudioSegment.from_file(source_file)

    # 파일 이름과 확장자를 분리합니다.
    base_name = os.path.splitext(os.path.basename(source_file))[0]

    # 오디오를 지정된 길이로 분할합니다.
    for i in range(0, len(audio), split_length):
        split_audio = audio[i:i+split_length]
        split_filename = f"{base_name}_part{i//split_length}.mp3"
        
        # 대상 폴더가 존재하지 않으면 생성합니다.
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # 분할된 오디오 파일을 저장합니다.
        split_audio.export(os.path.join(save_folder, split_filename), format='mp3')

def FFT_preprocessing(data, frame_rate):
    fft_values = fft(data)
    fft_freqs = fftfreq(len(data), 1/frame_rate)
    fft_magnitude = np.abs(fft_values)
    # Filter frequencies (0 Hz - 3000 Hz)
    freq_mask = (fft_freqs >= 0)
    filtered_freqs = fft_freqs[freq_mask]
    filtered_magnitude = fft_magnitude[freq_mask]

    # Visualize FFT results
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_freqs, filtered_magnitude)
    plt.title(f'FFT Magnitude (0 - 3000 Hz) for {filename}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.grid(True)
    #plt.xticks(np.arange(0, 3100, 100))
    plt.show()

    # Print the file name after processing
    print(f"Processed audio file: {audio_file_path}")
    
    return fft_values

def STFT_preprocessing(data, frame_rate, len_seg = 32768):
    # Perform STFT
    f, t, Zxx = stft(data, fs=frame_rate, nperseg=len_seg)

    # Frequency filtering (0 Hz - 2000 Hz)
    freq_index = np.where(f <= 100000)[0]
    f = f[freq_index]
    Zxx = Zxx[freq_index, :]

    # Calculate magnitude and apply threshold
    magnitude = np.abs(Zxx)

    # Visualize STFT results
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(t, f, magnitude, shading='gouraud', cmap='gray', norm=plt.Normalize(vmin=magnitude.min(), vmax=magnitude.max()))
    plt.title(f'STFT Magnitude for {filename}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    #plt.yticks(np.arange(0, 600, 100))
    plt.colorbar(label='Magnitude')
    plt.show()
    return f, t, Zxx

def mel_spectrogram_preprocessing(data, frame_rate, len_seg=10000, n_mels=128, fmax=1000):
    # Perform STFT
    f, t, Zxx = stft(data, fs=frame_rate, nperseg=len_seg)

    # Frequency filtering (0 Hz - 2000 Hz)
    freq_index = np.where(f <= fmax)[0]
    f = f[freq_index]
    Zxx = Zxx[freq_index, :]

    # Calculate magnitude
    magnitude = np.abs(Zxx)

    # Convert to Mel scale
    mel_spectrogram = librosa.feature.melspectrogram(S=magnitude, sr=frame_rate, n_mels=n_mels, fmax=fmax)

    # Convert to decibel scale (optional but common for visualization)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Visualize Mel Spectrogram
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(mel_spectrogram, sr=frame_rate, x_axis='time', y_axis='mel', fmax=fmax, cmap='gray')
    plt.title(f'Mel Spectrogram for {filename}')
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time [sec]')
    y_ticks = np.arange(0, 1001, 50)

    plt.yticks(y_ticks)
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    return mel_spectrogram, mel_spectrogram_db




# Directory containing the audio files
directory_path = r'C:\Users\SKT038\Desktop\data\sound\hornet'

# Process each audio file in the directory
for filename in os.listdir(directory_path):
    audio_file_path = os.path.join(directory_path, filename)
    data, fs = read_audio(audio_file_path)
    fft1 = STFT_preprocessing(data,fs)
    break
