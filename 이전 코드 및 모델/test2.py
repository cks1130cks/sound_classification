import requests
from io import BytesIO
import numpy as np
import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn.functional as F
import os
import logging
import warnings

# 경고 메시지 비활성화
warnings.simplefilter("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)



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


# 분류기 정의
class AudioClassifier(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(AudioClassifier, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

# 오디오 파일 로드 및 필터 적용
def load_audio(file_path, sr = 16000, cutoff=2000):
    signal, sr = librosa.load(file_path, sr=sr)
    filtered_signal = low_pass_filter(signal, sr, cutoff)
    return filtered_signal, sr

# 저주파 필터 함수
def low_pass_filter(audio, sr, cutoff=2000):
    # 푸리에 변환 수행
    audio_fft = np.fft.fft(audio)
    frequencies = np.fft.fftfreq(len(audio), 1/sr)
    
    # 0Hz ~ cutoff 주파수 범위 선택
    mask = (frequencies >= 0) & (frequencies <= cutoff)
    
    # 주파수 범위 내의 성분만 남기고, 나머지는 0으로 설정
    filtered_audio_fft = np.zeros_like(audio_fft)
    filtered_audio_fft[mask] = audio_fft[mask]
    
    # 역 푸리에 변환 수행
    filtered_audio = np.fft.ifft(filtered_audio_fft)
    
    # 실수부만 반환 (오차로 인해 소수의 허수부가 생길 수 있음)
    return np.real(filtered_audio)

# 오디오 데이터 처리
def process_audio(signal):
    inputs = processor(signal, return_tensors="pt", sampling_rate=16000)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        features = model(**inputs).last_hidden_state
    return features

# 예측 함수
def predict(audio_signal, sr):
    features = process_audio(audio_signal)
    features = features.mean(dim=1)  # 시퀀스 차원에 대한 평균
    outputs = classifier(features)
    # 로짓을 확률로 변환
    probabilities = F.softmax(outputs, dim=1)
    # 1번 레이블(예: 'hornet' 클래스)의 확률을 반환
    label_one_probability = probabilities[:, 0].item()
    return label_one_probability

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 모델 및 프로세서 로드
# model_name = "facebook/wav2vec2-large-960h-lv60-self"
# processor = Wav2Vec2Processor.from_pretrained(model_name)
# model = Wav2Vec2Model.from_pretrained(model_name).to(device)

# 이 파이썬 코드가 저장된 파일의 디렉토리 가져오기
script_dir = os.path.dirname(os.path.abspath(__file__))

# 로컬에 저장된 wav2vec2모델 사용
local_model_path = os.path.join(script_dir, "wav2vec2")
# 로컬에서 프로세서와 모델 로드
processor = Wav2Vec2Processor.from_pretrained(local_model_path)
model = Wav2Vec2Model.from_pretrained(local_model_path).to(device)


# 분류기 인스턴스 생성 및 모델 로드
num_classes = 2
feature_dim = model.config.hidden_size
classifier = AudioClassifier(feature_dim, num_classes).to(device)


# 로컬에 저장된 wav2vec2모델 사용
param = os.path.join(script_dir, "audio_classifier.pth")
# 모델을 CPU로 불러오기
classifier.load_state_dict(torch.load(param, map_location=torch.device('cpu')))





# 스트리밍 URL
stream_url = 'http://172.23.250.133:5000/audio'

# # 오디오 파일의 헤더 생성
# sample_rate = 16000  # 44.1kHz로 변경
# bits_per_sample = 16  # 16 bits
# channels = 2  # 스테레오
# wav_header = generate_wav_header(sample_rate, bits_per_sample, channels)

# # 누적된 데이터를 저장할 버퍼
# audio_buffer = BytesIO()
# audio_buffer.write(wav_header)  # 버퍼에 WAV 헤더 기록
# cnt = 0
# # 스트리밍 시작
# with requests.get(stream_url, stream=True) as r:
#     for chunk in r.iter_content(chunk_size=1024):
#         if chunk:
#             audio_buffer.write(chunk)  # 스트리밍 데이터를 버퍼에 기록

#             # 데이터를 누적하여 일정량 이상 쌓였을 때 디코딩 시도
#             if audio_buffer.tell() > len(wav_header) + (sample_rate * 2 * channels * bits_per_sample // 8 * 2):  # 2초 분량의 데이터가 쌓였을 때 시도
#                 try:
#                     audio_buffer.seek(0)
#                     # librosa로 오디오 데이터 로드
#                     audio, sr = librosa.load(audio_buffer, sr=sample_rate)
#                     hornet_prob = predict(audio,sr)
                    
#                     if hornet_prob > 0.3:
#                         # 0.3 이상일 때, 지정된 폴더에 오디오 파일 저장
#                         save_audio(audio, sr, folder_path=r"C:\Users\SKT038\Desktop\새 폴더", filename=f"{cnt}_hornet.wav")
#                         cnt+=1
#                         # 추가 처리 (예: 비전 모델로 신호 전송)
#                         # signal to vision model
#                         pass
#                     print(hornet_prob)
                    
#                     # 버퍼 초기화 및 헤더 재작성
#                     audio_buffer = BytesIO()
#                     audio_buffer.write(wav_header)
#                 except Exception as e:
#                     print(f"Error processing audio: {e}")
#                     continue

import httpx
mainserver_url="http://kulbul.iptime.org:8000/detector/"

async def audio_detect(stream_url):

    # stream_url = 'http://172.23.250.133:5000/audio'

    # 오디오 파일의 헤더 생성
    sample_rate = 16000  # 44.1kHz로 변경
    bits_per_sample = 16  # 16 bits
    channels = 1  # 스테레오
    wav_header = generate_wav_header(44100, bits_per_sample, channels)

    # 누적된 데이터를 저장할 버퍼
    audio_buffer = BytesIO()
    audio_buffer.write(wav_header)  # 버퍼에 WAV 헤더 기록
    cnt = 0
    # 스트리밍 시작
    with requests.get(stream_url, stream=True) as r:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                audio_buffer.write(chunk)  # 스트리밍 데이터를 버퍼에 기록

                # 데이터를 누적하여 일정량 이상 쌓였을 때 디코딩 시도
                if audio_buffer.tell() > len(wav_header) + (44100 * 4 * channels * bits_per_sample // 8):  # 2초 분량의 데이터가 쌓였을 때 시도
                    try:
                        audio_buffer.seek(0)
                        # librosa로 오디오 데이터 로드
                        audio, sr = load_audio(audio_buffer, sr=sample_rate)
                        hornet_prob = predict(audio,sr)
                        
                        if hornet_prob > 0.3:
                            url=mainserver_url+"detector/audio/detect/test"
                            async with httpx.AsyncClient() as client:
                                response = await client.get(url)
                            return True
                        print(hornet_prob)
                        
                        # 버퍼 초기화 및 헤더 재작성
                        audio_buffer = BytesIO()
                        audio_buffer.write(wav_header)
                    except Exception as e:
                        print(f"Error processing audio: {e}")
                        continue

