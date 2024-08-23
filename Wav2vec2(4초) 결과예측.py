import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import scipy.io.wavfile as wavfile
import os

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 모델 및 프로세서 로드
# model_name = "facebook/wav2vec2-large-960h-lv60-self"
# processor = Wav2Vec2Processor.from_pretrained(model_name)
# model = Wav2Vec2Model.from_pretrained(model_name).to(device)

# 로컬에 저장된 wav2vec2모델 사용
local_model_path = r"C:\Users\SKT038\Desktop\wav2vec2"
# 로컬에서 프로세서와 모델 로드
processor = Wav2Vec2Processor.from_pretrained(local_model_path)
model = Wav2Vec2Model.from_pretrained(local_model_path).to(device)




# 오디오 파일 로드 및 필터 적용
def load_audio(file_path, cutoff=2000):
    signal, sr = librosa.load(file_path, sr=16000)
    filtered_signal = low_pass_filter(signal, sr, cutoff)
    return filtered_signal, sr

# 저주파 필터 함수
def low_pass_filter(audio, sr, cutoff=2000):
    audio_fft = np.fft.rfft(audio)
    frequencies = np.fft.rfftfreq(len(audio), 1/sr)
    audio_fft[frequencies > cutoff] = 0
    filtered_audio = np.fft.irfft(audio_fft)
    return filtered_audio

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


# 오디오 신호를 4초 단위로 분할하는 함수
def split_audio(signal, sr, segment_length=4):
    segment_samples = sr * segment_length
    segments = []
    
    for i in range(0, len(signal), segment_samples):
        segment = signal[i:i + segment_samples]
        
        # 길이가 segment_samples보다 짧은 경우 0으로 패딩
        if len(segment) < segment_samples:
            padding = np.zeros(segment_samples - len(segment))
            segment = np.concatenate((segment, padding))
        
        segments.append(segment)
    
    return segments

# 특정 조건을 만족하는 오디오 데이터를 저장하는 함수
def save_audio(segment, sr, output_dir, index):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"segment_{index}.wav")
    wavfile.write(output_path, sr, (segment * 32767).astype(np.int16))  # 16-bit PCM 형식으로 저장

# 예측 함수
def predict(file_path, output_dir="output_segments"):
    audio_signal, sr = load_audio(file_path)
    segments = split_audio(audio_signal, sr)
    
    all_preds = []
    
    for i, segment in enumerate(segments):
        features = process_audio(segment)
        features = features.mean(dim=1)  # 시퀀스 차원에 대한 평균
        outputs = classifier(features)
        
        # 소프트맥스 함수를 통해 확률로 변환
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # # 라벨 1의 확률이 0.4 이하인 경우 오디오 세그먼트를 저장
        # if probabilities[0][1].item() >= 0.8:
        #     save_audio(segment, sr, output_dir, i)
        
        pred = torch.argmax(probabilities, dim=1)
        print(f"Probabilities: {probabilities}")
        
        all_preds.append(pred.item())
    
    return all_preds

# 예측 실행
file_path = r"C:\Users\SKT038\Desktop\test.wav"
preds = predict(file_path)
print(f"Predicted classes: {preds}")
