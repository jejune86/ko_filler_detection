import os
import librosa
import librosa.display
import pandas as pd
import numpy as np

# 경로 설정
csv_paths = ['fluencybank_labels.csv', 'SEP-28k_labels.csv']
audio_folder = 'clips'
save_folder = 'dataset'
os.makedirs(save_folder, exist_ok=True)

def get_audio_path(row):
    show = row['Show'].strip()
    clip_id = str(row['ClipId']).strip()

    if show == 'FluencyBank':
        ep_id = str(row['EpId']).zfill(3)
    else:  # 예: HeStutters
        ep_id = str(row['EpId'])  # zero padding 없이

    filename = f"{show}_{ep_id}_{clip_id}.wav"
    return os.path.join(audio_folder, filename)

def get_fluency_label(row):
    stutter_labels = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'DifficultToUnderstand', 'Interjection']
    fluent_labels = ['NoStutteredWords', 'NaturalPause', 'NoSpeech']

    is_stutter = sum([row[label] for label in stutter_labels])
    is_fluent = sum([row[label] for label in fluent_labels])

    if is_stutter > is_fluent:
        return 1  # 말 더듬
    else :
        return 0

# 멜스펙토그램 + 레이블 추출 함수
def extract_melspectrogram_and_label(row, sr=16000, n_mels=128, duration=3.0):
    audio_path = get_audio_path(row)

    try:
        y, _ = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"[오류] {audio_path} 로드 실패: {e}")
        return None, None
    

    # 너무 짧은 파일 스킵
    if len(y) < int(1 * sr):
        print(f"[건너뜀] {audio_path} - 길이 {len(y)/sr:.2f}초 (최소 {1}초 필요)")
        return None, None

    target_length = int(duration * sr)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    elif len(y) > target_length:
        y = y[:target_length]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    label = get_fluency_label(row)

    return mel_db, label

# 전체 CSV 데이터 로딩 및 통합
all_data = []
for path in csv_paths:
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

# 순차적으로 처리 및 저장
for idx, row in combined_df.iterrows():
    mel_spec, label = extract_melspectrogram_and_label(row)
    if mel_spec is None:
        continue

    filename = f"data_{idx+1:05d}.npz"
    save_path = os.path.join(save_folder, filename)
    np.savez(save_path, mel=mel_spec, label=label)
    print(f"[저장 완료] {save_path}")
