import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

class GrayScaleRoadClassifier(nn.Module):
    def __init__(self):
        super(GrayScaleRoadClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Change input channels to 1
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * (96 // 8) * (128 // 8), 128)  # Adjusted for pooled dimensions
        self.fc2 = nn.Linear(128, 3)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Convolutional layers with BatchNorm and LeakyReLU
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)
        x = self.fc2(x)  # No activation (logits)

        return x



def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, GrayScaleRoadClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'gray_scale_road_classifier.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))

def load_model():
    from torch import load
    from os import path
    r = GrayScaleRoadClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'gray_scale_road_classifier.th'), map_location='cpu'))
    return r
