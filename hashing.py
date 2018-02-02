#!/usr/bin/env python

import sys
import argparse
import binascii
import itertools
from functools import reduce, partial
from operator import add

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim

# There seems to be an unreasonably large amount of changes to the code to 
# target the GPU. This is not a pretty solution but better than 'if' statements
# around all the torch.FloatTenor creations.
CUDA = False
def cudize(in_):
    global CUDA
    return in_.cuda() if CUDA else in_


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
            self.r = nn.Parameter(data=cudize(torch.zeros(gen_length,
                raw_input_size)), requires_grad=True)

    def step(self, input_):
        return self.out(self.features(input_))

    def forward(self, input_gen, ignore_data_err=True):
        output = Variable(cudize(torch.zeros(self.output_size)))
        for index, chunk in enumerate(input_gen):
            input_ = Variable(cudize(torch.FloatTensor(chunk)))
            if not ignore_data_err:
                input_ += self.r[index]
            input_ = torch.cat((input_, output))
            output = self.step(input_)
        return output


class HashAttack(object):
    def __init__(self, model_class, target, source=None, output=None,
            adam_lr=1., loss_reg=0.0045):
        self._target = {'data': self.read_file(target)}
        self._source = {'data': self.read_file(source)} if source else None
        self._output_name = output
        self._adam_lr = adam_lr
        self._loss_reg = loss_reg
        if source is None:
            self._model = cudize(model_class(data_err=False, 
                gen_length=len(self._target['data'])))
        else:
            self._model = cudize(model_class(data_err=True, 
                gen_length=len(self._source['data'])))
            self._source['output'] = Variable(self._model(self._source['data'],
                ignore_data_err=True).data, requires_grad=False)
            self._source['param'] = nn.Parameter(
                    cudize(torch.FloatTensor(self._source['data'])))
        self._target['output'] = Variable(self._model(self._target['data'],
            ignore_data_err=True).data, requires_grad=False)

    @staticmethod
    def read_file(file_name, chunksize=64): # chunksize is in bytes
        data = []
        with open(file_name, 'rb') as fp:
            while True:
                bytes_ = fp.read(chunksize)
                if bytes_:
                    data.append(HashAttack.expand_bytes(
                        bytes_.ljust(chunksize, b'\0')))
                else:
                    break
        return data

    @staticmethod
    def expand_bytes(bytes_):
        return reduce(add,
                [b'\x01' if (byte & 1<<bit) else b'\x00' \
                        for byte in bytes_ \
                        for bit in reversed(range(8))])

    @staticmethod
    def report_hash(model_output, label='', **kwargs):
        raw_list = model_output.data.cpu().numpy().tolist()
        f = lambda i: str(int(i + .5))
        o = hex(int(reduce(add, map(f, raw_list)), 2))
        print('{} - {}'.format(o, label), **kwargs)

    @staticmethod
    def check_eq(first, second):
        f = lambda i: str(int(i + .5))
        o_first = hex(int(reduce(add, map(f, first)), 2))
        o_second = hex(int(reduce(add, map(f, second)), 2))
        return o_first == o_second

    @staticmethod
    def report_successful(input_, r):
        data = (input_ + r).data.cpu().numpy()
        low, high = data[data < .5], data[data >= .5]
        print('Low mean: {:.4f}, std: {:.4f}, min: {:.4f}, max: {:.4f}'.format(
            np.mean(low), np.std(low), np.min(low), np.max(low)))
        print('High mean: {:.4f}, std: {:.4f}, min: {:.4f}, max: {:.4f}'.format(
            np.mean(high), np.std(high), np.min(high), np.max(high)))

    def write_output(self, r):
        if self._output_name is None:
            return
        data = ((self._source['param'] + r).data.cpu().numpy() >= .5)
        data = data.flatten().astype(int).tolist()
        o = hex(int(reduce(add, map(str, data)), 2))
        with open(self._output_name, 'wb') as fout:
            fout.write(binascii.unhexlify(o[2:]))

    # W function with zeros as 0 and 1, the two valid binary inputs
    def loss_reg(self, r):
        x = self._source['param'] + r
        ret = torch.mean(torch.pow(x * (x - 1), 2))
        return ret

    def query(self, target=None):
        if target is None:
            target_output = self._target['output']
        else:
            target = self.read_file(target)
            target_output = self._model(target, ignore_data_err=True)
        self.report_hash(target_output, 'target hash')

    def iterate(self, steps=0):
        self.report_hash(self._source['output'], 'original source hash')
        self.report_hash(self._target['output'], 'target hash')
        mse_loss = nn.MSELoss()
        optimizer = optim.Adam(params=[self._model.r], lr=self._adam_lr)

        success = False
        for iteration in range(steps) if steps else itertools.count():
            optimizer.zero_grad()
            outputs = self._model(self._source['data'], ignore_data_err=False)
            loss = mse_loss(outputs, self._target['output'])
            adv_loss = loss + self._loss_reg * self.loss_reg(self._model.r)
            if self.check_eq(self._target['output'], outputs):
                success = True
                self.report_hash(outputs, 'step... {} *'.format(iteration))
                self.report_successful(self._source['param'], self._model.r)
                self.write_output(self._model.r)
                break
            if iteration % 10 == 0:
                self.report_hash(outputs, 'step... {}'.format(iteration),
                        end='\r')
            adv_loss.backward()
            optimizer.step()

        print('\n...{}'.format('success' if success else 'failure'))


if __name__ == '__main__': # expose variables to ipython
    # Without this experiments are not repeatable!
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description='Attack RNN hasher.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('target', type=str,
            help='target file for target hash value')
    parser.add_argument('--source', type=str, default=None,
            help='source file to be transformed to have target hash')
    parser.add_argument('--output', type=str, default=None,
            help='source file to be transformed to have target hash')
    parser.add_argument('--adam-lr', type=float, default=1.,
            help='learning rate for adam optimizer')
    parser.add_argument('--loss-reg', type=float, default=0.0045,
            help='coefficient for loss regularization')
    parser.add_argument('--iters', type=int, default=0,
            help='number of steps (default is infinite loop)')
    parser.add_argument('--cuda', action='store_true',
            help='use GPU or not')
    args = parser.parse_args()

    CUDA = args.cuda
    att = HashAttack(partial(RNN, 512, [1204, 2048], 128), args.target, 
            source=args.source, output=args.output, adam_lr=args.adam_lr, 
            loss_reg=args.loss_reg)
    if args.source:
        att.iterate()
    else:
        att.query()
