import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

class CNN(nn.Module):
    def __init__(self, raw_input_size, feature_size, output_size, cudize,
            data_err=False):
        super().__init__()
        self._raw_input_size = raw_input_size
        self._feature_size = feature_size
        self._cudize = cudize
        self._data_err = data_err

        self._conv = nn.Sequential(
                nn.Conv2d(1, feature_size, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(feature_size, feature_size, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2))
        self._fc = nn.Sequential(
                nn.Linear(feature_size * 4 * 4, 64),
                nn.ReLU(),
                nn.Linear(64, output_size),
                nn.LogSoftmax())
        if data_err:
            self._data_delta = Parameter(
                    data=self._cudize(torch.zeros(raw_input_size)))

    def cudize(self):
        return self._cudize(self)

    @property
    def r(self):
        return self._data_delta

    def forward(self, input_, ignore_data_err=True):
        if not ignore_data_err:
            input_ += self._data_delta
            input_ = torch.clamp(input_, 0, 1)
        conv = self._conv(input_)
        fc_input = conv.view(-1, self._feature_size * 4 * 4)
        output = self._fc(fc_input)
        return output
