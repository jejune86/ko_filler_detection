import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)


class CNN_RNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 80, T)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),                        # (B, 32, 40, T/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, 40, T/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))                         # (B, 64, 20, T/4)
        )

        self.project = nn.Linear(64 * 20, 64)  # 64 channels × 20 height → 64 dims
        self.rnn = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(64 * 2, num_classes)

    def forward(self, x):  # x: (B, 1, 80, T)
        x = self.cnn(x)    # (B, 64, 20, T')
        x = x.permute(0, 3, 1, 2)  # (B, T', C, H)
        x = x.flatten(2)           # (B, T', C*H)
        x = self.project(x)        # (B, T', 64)
        out, _ = self.rnn(x)       # (B, T', 128)
        x = out.mean(dim=1)        # (B, 128)
        return self.classifier(x)  # (B, 3)