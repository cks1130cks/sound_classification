import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from pydub import AudioSegment
import os
import matplotlib as mpl

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
    return audio.frame_rate, data

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
    fft_freqs = fftfreq(len(data), 1/frame_ratefs)
    fft_magnitude = np.abs(fft_values)
    return fft_values,fft_freqs,fft_magnitude

def STFT_preprocessing(data, frame_rate, len_seg):
    frequency, t, Zxx = stft(data,fs = frame_rate, nperseg = len_seg)
    return frequency, t, Zxx

def frequency_filtering(freq, Lower_Bound, Upper_Bound):
    freq_index = np.where(Lower_Bound<= freq <= Upper_Bound)[0]
    freq = freq[freq_index]
    return freq

