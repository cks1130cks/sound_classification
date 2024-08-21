import requests
import time
from io import BytesIO
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import io
from pydub.exceptions import CouldntDecodeError



def generate_wav_header(sample_rate, bits_per_sample, channels):
    datasize = 2000 * 10**6
    o = bytes("RIFF", 'ascii')
    o += (datasize + 36).to_bytes(4, 'little')
    o += bytes("WAVE", 'ascii')
    o += bytes("fmt ", 'ascii')
    o += (16).to_bytes(4, 'little')
    o += (1).to_bytes(2, 'little')
    o += (channels).to_bytes(2, 'little')
    o += (sample_rate).to_bytes(4, 'little')
    o += (sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little')
    o += (channels * bits_per_sample // 8).to_bytes(2, 'little')
    o += (bits_per_sample).to_bytes(2, 'little')
    o += bytes("data", 'ascii')
    o += (datasize).to_bytes(4, 'little')
    return o

# 예시 클래스 레이블과 레이블 인코더 설정
class_labels = ['hornet', 'not_hornet']
le = LabelEncoder()
le.fit(class_labels)

# 모델 불러오기
model_path = r'C:\Users\SKT038\Desktop\test\Model3.h5'
model = load_model(model_path)


# 스트리밍 URL
stream_url = 'http://172.23.250.133:5000/audio'

# 오디오 파일의 헤더 생성
sample_rate = 44100  # 44.1kHz
bits_per_sample = 16  # 16 bits
channels = 2  # 스테레오
wav_header = generate_wav_header(sample_rate, bits_per_sample, channels)


# 누적된 데이터를 저장할 버퍼
audio_buffer = BytesIO()
audio_buffer.write(wav_header)  # 버퍼에 WAV 헤더 기록


# 스트리밍 시작
with requests.get(stream_url, stream=True) as r:
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:
            audio_buffer.write(chunk)  # 스트리밍 데이터를 버퍼에 기록

            # 데이터를 누적하여 일정량 이상 쌓였을 때 디코딩 시도
            if audio_buffer.tell() > len(wav_header) + 44100 * 4:  # 4초 분량의 데이터가 쌓였을 때 시도
                try:
                    audio_buffer.seek(0)
                    # libosa로 오디오 데이터 로드
                    audio, sr = librosa.load(audio_buffer, sr=None)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)
                    mfccs_processed = np.mean(mfccs.T, axis=0)
                    mfccs_processed = mfccs_processed.reshape(16, 8, 1)
                    # 예측 확률 계산
                    prediction = model.predict(np.array([mfccs_processed]))
                    hornet_prob = prediction[0][le.transform(['hornet'])[0]]
                    if hornet_prob>0.3:
                        # signal to vision model
                        pass
                    print(round(hornet_prob, 3))
                    
                    # 버퍼 초기화 및 헤더 재작성
                    audio_buffer = BytesIO()
                    audio_buffer.write(wav_header)
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    continue
