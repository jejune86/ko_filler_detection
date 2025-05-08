import os
import librosa
import torch
from tqdm import tqdm
import numpy as np


LABEL_MAP = {"filler": 0, "stutter": 1, "fluent": 2}
SAVE_DIR = "mel_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

def preprocess_one(file_path, label):
    y, sr = librosa.load(file_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return torch.tensor(mel_norm).unsqueeze(0), label


def preprocess_all():
    for label_name, label_id in LABEL_MAP.items():
        folder = os.path.join("dataset", label_name)
        for idx, fname in enumerate(tqdm(os.listdir(folder))):
            if fname.endswith(".wav"):
                path = os.path.join(folder, fname)
                mel, label = preprocess_one(path, label_id)
                save_name = f"{label_name}_{idx:03d}.pt"  # stutter_000.pt 등
                torch.save({"mel": mel, "label": label}, os.path.join(SAVE_DIR, save_name))

if __name__ == "__main__":
    preprocess_all()
