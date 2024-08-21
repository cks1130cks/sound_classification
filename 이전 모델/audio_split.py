from pydub import AudioSegment
import os

def split_and_pad_audio(base_folder, source_folder, target_folder, split_length=4000):
    # 전체 경로를 계산합니다.
    full_source_path = os.path.join(base_folder, source_folder)
    full_target_path = os.path.join(base_folder, target_folder)

    # full_source_path에서 오디오 파일을 찾아 처리합니다.
    for subdir, dirs, files in os.walk(full_source_path):
        for file in files:
            if file.endswith('.mp3') or file.endswith('.wav'):
                audio_path = os.path.join(subdir, file)
                audio = AudioSegment.from_file(audio_path)
                
                # 오디오를 4초 단위로 분할합니다.
                for i in range(0, len(audio), split_length):
                    split_audio = audio[i:i+split_length]
                    
                    # 4초보다 짧으면 무음으로 패딩
                    if len(split_audio) < split_length:
                        padding = AudioSegment.silent(duration=split_length - len(split_audio))
                        split_audio += padding
                    
                    base_name = os.path.splitext(file)[0]
                    split_filename = f"{base_name}_part{i//split_length}.wav"
                    
                    # 대상 폴더의 구조를 생성합니다.
                    target_subdir = subdir.replace(full_source_path, full_target_path)
                    if not os.path.exists(target_subdir):
                        os.makedirs(target_subdir)
                    
                    # 분할된 오디오 파일을 저장합니다.
                    split_audio.export(os.path.join(target_subdir, split_filename), format='wav')

# 사용 예시
base_folder = r'C:\Users\SKT038\Desktop\sound_data\sound\test'  # 원하는 기본 폴더의 전체 경로
source_folders = ['hornet', 'not_hornet']
target_folders = ['hornet_1', 'not_hornet_1']
for source, target in zip(source_folders, target_folders):
    split_and_pad_audio(base_folder, source, target)
