#!/usr/bin/env python

import sys
from functools import reduce
from operator import add

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim


class RNN(nn.Module):
    def __init__(self, raw_input_size, hidden_sizes, output_size):
        super(RNN, self).__init__()
        self.raw_input_size = raw_input_size
        self.output_size = output_size

        act = nn.Tanh()
        last_size = raw_input_size + output_size

        sequentials = []
        for hsize in hidden_sizes:
            sequentials.append(nn.Linear(last_size, hsize))
            sequentials.append(act)
            last_size = hsize

        sequentials.append(nn.Linear(last_size, output_size))

        self.features = nn.Sequential(*sequentials)
        self.out = nn.Sigmoid()

    def step(self, input_):
        return self.out(self.features(input_))

    def forward(self, input_gen):
        output = Variable(torch.zeros(self.output_size))
        for chunk in input_gen:
            input_ = torch.cat((Variable(torch.FloatTensor(chunk) / 256), output))
            output = self.step(input_)
        return output


def file_gen(file_name, chunksize):
    with open(file_name, 'rb') as fp:
        while True:
            bytes_ = fp.read(chunksize)
            if bytes_:
                yield bytes_.ljust(chunksize, b'\0')
            else:
                break

def main():
    torch.manual_seed(42)
    model = model = RNN(64, [1204, 2048], 16)
    g = file_gen(sys.argv[1], 64)

    output = model(g).data.cpu().numpy().tolist()
    f = lambda i: hex(min(255, max(0, int(256 * i))))[2:]
    print('{}'.format( reduce(add, map(f, output)) ))

if __name__ == '__main__':
    main()
