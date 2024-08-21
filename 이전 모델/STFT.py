import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from pydub import AudioSegment
import os
import matplotlib as mpl

# 한글 지원 글꼴로 변경
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
def read_audio(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension in ['.mp3', '.wav', '.flv', '.mp4', '.ogg', '.flac', '.m4a']:
        audio = AudioSegment.from_file(file_path)
        data = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            if len(data) % 2 != 0:
                data = data[:-1]
            data = data.reshape((-1, 2))
            data = data.mean(axis=1)
        return audio.frame_rate, data
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")

# Directory path
directory_path = 'C:/Users/SKT038/Desktop/유튜브/말벌'
for filename in os.listdir(directory_path):
    audio_file_path = os.path.join(directory_path, filename)

    # Read audio file
    fs, data = read_audio(audio_file_path)

    # Perform STFT
    f, t, Zxx = stft(data, fs=fs, nperseg=10000)

    # Frequency filtering (0 Hz - 2000 Hz)
    freq_index = np.where(f <= 500)[0]
    f = f[freq_index]
    Zxx = Zxx[freq_index, :]

    # Calculate magnitude and apply threshold
    magnitude = np.abs(Zxx)
    mean_val = np.mean(magnitude)
    variance = np.var(magnitude)
    threshold = mean_val + 2 * variance

    # Clip values exceeding the threshold
    magnitude = np.where(magnitude > threshold, threshold, magnitude)

    # Visualize STFT results
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(t, f, magnitude, shading='gouraud', cmap='gray', norm=plt.Normalize(vmin=magnitude.min(), vmax=magnitude.max()))
    plt.title(f'STFT Magnitude for {filename}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.yticks(np.arange(0, 600, 100))
    plt.colorbar(label='Magnitude')

    # Save the plot
    plt.savefig(f'C:/Users/SKT038/Desktop/유튜브/STFT/말벌/{os.path.splitext(filename)[0]}_stft.png')
    plt.close()

print("모든 STFT 그래프가 0~2000 Hz 범위에서 저장되었습니다.")
