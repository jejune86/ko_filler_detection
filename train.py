import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# 1. Dataset 클래스 정의
class MelDataset(Dataset):
    def __init__(self, folder_path):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        mel = data['mel']  # (n_mels, time)
        label = int(data['label'])

        mel = torch.tensor(mel).unsqueeze(0).float()  # (1, n_mels, time)
        label = torch.tensor(label).long()

        return mel, label

# 2. CNN 모델 정의 (간단한 구조)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # -> (B, 16, 128, 94)
        self.pool1 = nn.MaxPool2d(2)                              # -> (B, 16, 64, 47)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # -> (B, 32, 64, 47)
        self.pool2 = nn.MaxPool2d(2)                              # -> (B, 32, 32, 23)

        self.fc1 = nn.Linear(32 * 32 * 23, 64)  # Flatten 후
        self.fc2 = nn.Linear(64, 2)             # 이진 분류 (stutter or not)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))   # (B, 16, 64, 47)
        x = self.pool2(F.relu(self.conv2(x)))   # (B, 32, 32, 23)
        x = x.view(x.size(0), -1)               # Flatten
        x = F.relu(self.fc1(x))                 # (B, 64)
        return self.fc2(x)                      # (B, 2)

# 3. 데이터 로드
dataset = MelDataset('dataset')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# 4. 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. 학습 루프
for epoch in range(10):
    model.train()
    total_loss = 0
    print(f"\n[Epoch {epoch+1}]")

    for mel, label in tqdm(train_loader, desc="훈련 진행중"):
        mel, label = mel.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(mel)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 검증
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for mel, label in val_loader:
            mel, label = mel.to(device), label.to(device)
            output = model(mel)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

    acc = correct / total * 100
    print(f"Train Loss: {total_loss:.4f}, Val Accuracy: {acc:.2f}%")