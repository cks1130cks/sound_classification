from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch

# 분류기 정의
class AudioClassifier(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(AudioClassifier, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 모델 및 프로세서 로컬에 저장 (최초 1회 실행)
model_name = "facebook/wav2vec2-large-960h-lv60-self"
local_directory = r"C:\Users\SKT038\Desktop\Wav2vec2\local_wav2vec2_model"

# 로컬에 모델 저장
processor = Wav2Vec2Processor.from_pretrained(model_name)
processor.save_pretrained(local_directory)

model = Wav2Vec2Model.from_pretrained(model_name)

# 분류기 인스턴스 생성 및 모델 로드
num_classes = 2
feature_dim = model.config.hidden_size
classifier = AudioClassifier(feature_dim, num_classes).to(device)

# 모델을 CPU로 불러오기
classifier.load_state_dict(torch.load(r"C:\Users\SKT038\Desktop\sound_classification\audio_classifier.pth", map_location=torch.device('cpu')))


# 수정된 모델 저장
model.save_pretrained(local_directory)