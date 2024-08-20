import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


# 예시 클래스 레이블
class_labels = ['hornet', 'not_hornet']  # 'not_hornet' 오타 수정
# LabelEncoder 인스턴스 생성 및 학습
le = LabelEncoder()
le.fit(class_labels)

# HDF5 형식으로 저장된 모델 파일 경로
model_path = r'C:\Users\SKT038\Desktop\Model3.h5'

# 모델 불러오기
model = load_model(model_path)

def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    # MFCC 특징 추출
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)
    # 평균을 사용하여 2D MFCC를 1D 벡터로 축소
    mfccs_processed = np.mean(mfccs.T, axis=0)
    mfccs_processed = mfccs_processed.reshape(16, 8, 1)
    return mfccs_processed

def process_audio(input_source):
    # 오디오 파일 불러오기
    processed_audio = preprocess_audio(input_source)
    
    prediction_probabilities = model.predict(np.array([processed_audio]))
    
    return prediction_probabilities

def is_hornet(input_source, threshold=0.2):
    # 예측 확률 계산
    prob = process_audio(input_source)
    
    # hornet 클래스의 확률을 가져오기 (클래스 레이블이 'hornet'인 경우)
    hornet_prob = prob[0][le.transform(['hornet'])[0]]
    
    # 확률이 threshold 이상인 경우 True, 그렇지 않은 경우 False
    return hornet_prob >= threshold

# 예제 파일에 대해 함수 호출
result = is_hornet(r'C:\Users\SKT038\Desktop\sound_data\split_sound\hornet\[60FPS HQ 다큐] 장수말벌의 침입을 막는 꿀벌의 처절한 사투 - 320-1_part0_shift.wav')
print(result)
