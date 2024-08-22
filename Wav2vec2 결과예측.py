import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 프로세서 로드
model_name = "facebook/wav2vec2-large-960h-lv60-self"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name).to(device)

# 오디오 파일 로드 및 필터 적용
def load_audio(file_path, cutoff=2000):
    signal, sr = librosa.load(file_path, sr=16000)
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

# 분류기 정의
class AudioClassifier(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(AudioClassifier, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

# 분류기 인스턴스 생성 및 모델 로드
num_classes = 2
feature_dim = model.config.hidden_size
classifier = AudioClassifier(feature_dim, num_classes).to(device)
# 모델을 CPU로 불러오기
classifier.load_state_dict(torch.load(r"C:\Users\SKT038\Desktop\sound_classification\audio_classifier.pth", map_location=torch.device('cpu')))


# 예측 함수
def predict(file_path):
    audio_signal, sr = load_audio(file_path)
    features = process_audio(audio_signal)
    features = features.mean(dim=1)  # 시퀀스 차원에 대한 평균
    outputs = classifier(features)
    pred = torch.argmax(outputs, dim=1)
    return pred.item()

# 예측 실행
file_path = r"C:\Users\SKT038\Desktop\test.wav"
pred = predict(file_path)
print(f"Predicted class: {pred}")