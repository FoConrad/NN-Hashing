import abc
import sys
import itertools
import pickle 
from functools import reduce
from operator import add

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class MNISTAttack(object):
    def __init__(self, model_class, weights_file):
        self._model_class = model_class
        self._model = self.load_target_model(model_class, weights_file)

    @staticmethod
    def load_target_model(model_class, weights_file):
        model = model_class(data_err=True)
        weights = {}
        try:
            with open(weights_file, 'rb') as f:
                weights = pickle.load(f)
        except FileNotFoundError as fe:
            print('Excretions encountered:', fe)
            print('Please execute "{}" to train target model'.format(
                'python -m attacks.mnist_base'))
            sys.exit(-1)
        for param in model.named_parameters():
            if param[0] in weights.keys():
                param[1].data = weights[param[0]].data
        return model
