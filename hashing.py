#!/usr/bin/env python

import sys
import argparse
import binascii
import itertools
from functools import reduce
from operator import add

import numpy as np
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

def finish_succesful(input_, r):
    input_ = nn.Parameter(torch.FloatTensor(input_))
    data = (input_ + r).data.numpy()
    low, high = data[data < .5], data[data >= .5]
    print('Low mean: {:.4f}, std: {:.4f}, min: {:.4f}, max: {:.4f}'.format(np.mean(low),
        np.std(low), np.min(low), np.max(low)))
    print('High mean: {:.4f}, std: {:.4f}, min: {:.4f}, max: {:.4f}'.format(np.mean(high),
        np.std(high), np.min(high), np.max(high)))

# W function with zeros as 0 and 1, the two valid binary inputs
def my_loss(data, r):
    data = nn.Parameter(torch.FloatTensor(data))
    x = data + r
    ret = torch.mean(torch.pow(x * (x - 1), 2))
    return ret

def write_output(input_, r, output_name):
    input_ = nn.Parameter(torch.FloatTensor(input_))
    data = ((input_ + r).data.numpy() >= .5).flatten().astype(int).tolist()
    o = hex(int(reduce(add, map(str, data)), 2))
    with open(output_name, 'wb') as fout:
        fout.write(binascii.unhexlify(o[2:]))

if __name__ == '__main__': # expose variables to ipython
    # Without this experiments are not repeatable!
    torch.manual_seed(42)

    # Argument parsing
    parser = argparse.ArgumentParser(description='Attack RNN hasher.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('target', type=str, help='target file for target hash value')
    parser.add_argument('--source', type=str, default=None,
            help='source file to be transformed to have target hash')
    parser.add_argument('--output', type=str, default=None,
            help='source file to be transformed to have target hash')
    parser.add_argument('--adam-lr', type=float, default=1.,
            help='learning rate for adam optimizer')
    parser.add_argument('--loss-reg', type=float, default=0.0045,
            help='coefficient for loss regularization')
    parser.add_argument('--iters', type=int, default=0,
            help='how many iterations to optimize for (default is infinite loop)')
    args = parser.parse_args()

    # Simple checking
    if args.source is None:
        target_data = file_gen(args.target, 512 // 8)
        model = model = RNN(512, [1204, 2048], 128, data_err=True,
                gen_length=len(target_data))
        target_output = model(target_data, ignore_data_err=True)
        print_hash(target_output, 'target hash')
        sys.exit(0)

    # Define model and target
    source_data = file_gen(args.source, 512 // 8)
    model = model = RNN(512, [1204, 2048], 128, data_err=True,
            gen_length=len(source_data))

    target_output = model(file_gen(args.target, 512 // 8), ignore_data_err=True)
    target_output = Variable(target_output.data, requires_grad=False)
    source_orig = model(source_data, ignore_data_err=True)
    print_hash(source_orig, 'original source hash')
    print_hash(target_output, 'target hash')

    softmaxwithxent = nn.MSELoss()
    optimizer = optim.Adam(params=[model.r], lr=args.adam_lr)

    success = False
    for iteration in range(args.iters) if args.iters else itertools.count():
        optimizer.zero_grad()
        outputs = model(source_data, ignore_data_err=False)

        xent_loss = softmaxwithxent(outputs, target_output)
        adv_loss = xent_loss + args.loss_reg * my_loss(source_data, model.r)

        if equality(target_output, outputs):
            success = True
            print_hash(outputs, 'SUCCESS {} *'.format(iteration), end='\r')
            print('\n..success')
            finish_succesful(source_data, model.r)
            if args.output is not None:
                try:
                    write_output(source_data, model.r, args.output)
                except:
                    sys.exit(1)
            break

        if iteration % 10 == 0:
            print_hash(outputs, 'step... {}'.format(iteration), end='\r')

        adv_loss.backward()
        optimizer.step()

    # Done
    if not success:
        print('\n...failed')
