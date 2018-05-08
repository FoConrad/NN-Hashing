import abc
import binascii
import itertools
from functools import reduce
from operator import add

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class HashBaseAttack(object, metaclass=abc.ABCMeta):
    def __init__(self, model_class, target, source=None, output=None):
        self._target = {'data': self.read_file(target)}
        self._source = {'data': self.read_file(source)} if source else None
        self._output_name = output
        if source is None:
            self._model = model_class(
                gen_length=len(self._target['data'])).cudize()
        else:
            self._model = model_class(
                gen_length=len(self._source['data'])).cudize()
            self._source['output'] = Variable(self._model(self._source['data'],
                ignore_data_err=True).data, requires_grad=False)
            self._source['param'] = nn.Parameter(
                self._model._cudize(torch.FloatTensor(self._source['data'])))
        self._target['output'] = Variable(self._model(self._target['data'],
            ignore_data_err=True).data, requires_grad=False)

    @staticmethod
    def read_file(file_name, chunksize=64): # chunksize is in bytes
        data = []
        with open(file_name, 'rb') as fp:
            while True:
                bytes_ = fp.read(chunksize)
                if bytes_:
                    data.append(HashBaseAttack.expand_bytes(
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
        try:
            with open(self._output_name, 'wb') as fout:
                fout.write(binascii.unhexlify(o[2:]))
        except:
            pass

    def query(self, target=None):
        if target is None:
            target_output = self._target['output']
        else:
            target = self.read_file(target)
            target_output = self._model(target, ignore_data_err=True)
        self.report_hash(target_output, 'target hash')

    @abc.abstractmethod
    def loss_reg(self, r):
        return None

    @abc.abstractmethod
    def iterate(self, steps=0):
        pass
