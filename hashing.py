#!/usr/bin/env python

import sys
import argparse
import binascii
from functools import reduce
from operator import add

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim


class RNN(nn.Module):
    def __init__(self, raw_input_size, hidden_sizes, output_size, 
            data_err=False, gen_length=0):
        super(RNN, self).__init__()
        self.raw_input_size = raw_input_size
        self.output_size = output_size
        self.data_err = data_err

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

        if data_err:
            self.r = nn.Parameter(data=torch.zeros(gen_length,raw_input_size), 
                    requires_grad=True) # really small initial values

    def step(self, input_):
        return self.out(self.features(input_))

    def forward(self, input_gen, ignore_data_err=True):
        output = Variable(torch.zeros(self.output_size))
        for index, chunk in enumerate(input_gen):
            input_ = Variable(torch.FloatTensor(chunk))
            if not ignore_data_err:
                input_ += self.r[index]
            input_ = torch.cat((input_, output))
            output = self.step(input_)
        return output

def expand_bytes(bytes_):
    return reduce(add, 
            [b'\x01' if (byte & 1<<bit) else b'\x00' \
                    for byte in bytes_ \
                    for bit in reversed(range(8))])

def file_gen(file_name, chunksize):
    ret = []
    with open(file_name, 'rb') as fp:
        while True:
            bytes_ = fp.read(chunksize)
            if bytes_:
                ret.append(expand_bytes(bytes_.ljust(chunksize, b'\0')))
            else:
                break
    return ret

def print_hash(model_output, label='', **kwargs):
    raw_list = model_output.data.cpu().numpy().tolist()
    f = lambda i: str(int(i + .5))
    o = hex(int(reduce(add, map(f, raw_list)), 2))
    print('{} - {}'.format(o, label), **kwargs)

def equality(first, second):
    f = lambda i: str(int(i + .5))
    o_first = hex(int(reduce(add, map(f, first)), 2))
    o_second = hex(int(reduce(add, map(f, second)), 2))
    return o_first == o_second


if __name__ == '__main__': # expose variables to ipython
    # Without this experiments are not repeatable!
    torch.manual_seed(42)

    # Argument parsing
    parser = argparse.ArgumentParser(description='Attack RNN hasher.')
    parser.add_argument('target', type=str, help='target file for target hash value')
    parser.add_argument('--source', type=str, default=None,
            help='source file to be transformed to have target hash')
    parser.add_argument('--output', type=str, default='hash_me',
            help='source file to be transformed to have target hash')
    args = parser.parse_args()

    # Define model and target
    source_data = file_gen(args.source, 512 // 8)
    model = model = RNN(512, [1204, 2048], 128, data_err=True, 
            gen_length=len(source_data))

    target_output = model(file_gen(args.target, 512 // 8), ignore_data_err=True)
    target_output = Variable(target_output.data, requires_grad=False)
    source_orig = model(source_data, ignore_data_err=True)
    print_hash(target_output, 'target hash')
    print_hash(source_orig, 'original source hash')

    if args.source is None:
        sys.exit(0)

    softmaxwithxent = nn.MSELoss()
    optimizer = optim.Adam(params=[model.r], lr=1)

    for iteration in range(10000):
        optimizer.zero_grad()
        outputs = model(source_data, ignore_data_err=False)

        xent_loss = softmaxwithxent(outputs, target_output)
        adv_loss = xent_loss

        if equality(target_output, outputs):
            print('\nWE DID IT')
            break

        if iteration % 10 == 0:
            print_hash(outputs, 'step... {}'.format(iteration), end='\r')

        adv_loss.backward()
        optimizer.step()

    # Done
    print(' '*40 + '.....done\n')
