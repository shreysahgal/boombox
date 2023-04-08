import torch
import torch.nn as nn
import torch.nn.functional as F

class GenreClassifier(nn.Module):
    def __init__(self):
        super(GenreClassifier, self).__init__()
        self.conv1 = nn.Conv1d(10, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.AvgPool1d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 36, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        print(x.shape)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        print(x.shape)
        x = x.view(-1, 256 * 36)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
