import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _calc_fc1_in_features(input_sz, channels_before_flatten):
    def _calc_conv_osize(sz, k, s, pad):
        return math.floor((sz + 2*pad - k) / s) + 1
    isz = input_sz
    isz = _calc_conv_osize(isz, 8, 4, 0)
    isz = _calc_conv_osize(isz, 4, 2, 0)
    isz = _calc_conv_osize(isz, 3, 1, 0)
    return isz * isz * channels_before_flatten


class DQN(nn.Module):
    def __init__(self, num_actions=18, image_shape=(4, 84, 84)):
        super().__init__()

        self.num_actions = num_actions
        self.image_shape = image_shape
        input_c, input_h, input_w = image_shape
        assert input_h == input_w, "input image must be square"
        fc1_in_features = _calc_fc1_in_features(input_h, 64)

        self.conv1 = nn.Conv2d(input_c, 32, kernel_size=8, stride=4, padding=0, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(fc1_in_features, 512, bias=True)
        self.fc2 = nn.Linear(512, num_actions, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = DQN()
dummy_data = torch.randn(1, *model.image_shape)
model(dummy_data)
