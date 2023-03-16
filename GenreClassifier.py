import torch
import torch.nn as nn

class GenreClassifier(nn.Module):
    def __init__(self, classes=5):
        super(GenreClassifier, self).__init__()

        self.conv1 = nn.Conv1d(10, 8, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AvgPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.AvgPool1d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=0.3)

        self.fc = nn.Linear(2816, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = self.flatten(x)

        x = self.dropout(x)

        x = self.fc(x)
        x = self.softmax(x)

        return x
