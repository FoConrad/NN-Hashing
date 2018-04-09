import abc
import time
import sys
import itertools
import pickle 
import random
from functools import reduce
from operator import add

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class MNISTAttack(object, metaclass=abc.ABCMeta):
    def __init__(self, model_class, weights_file):
        self._model = self.load_target_model(model_class, weights_file)
        self._loss_fn = nn.CrossEntropyLoss()
        self._optimizer = optim.SGD(params=[self._model.r], lr=8e-3)

    @staticmethod
    def load_target_model(model_class, weights_file):
        model = model_class(data_err=True)
        weights = {}
        try:
            with open(weights_file, 'rb') as f:
                weights = pickle.load(f)
        except FileNotFoundError as fe:
            print('Exception encountered in MNISTAttack.load_target_model:', fe)
            print('Please execute "{}" to train target model'.format(
                'python adversarial_images.py --setup'))
            sys.exit(-1)
        for param in model.named_parameters():
            if param[0] in weights.keys():
                param[1].data = weights[param[0]].data
        return model

    @staticmethod
    def _get_samples(filepath):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError as fe:
            print('Exception encountered in MNISTAttack._get_samples:', fe)
            print('Please execute "{}" to train target model'.format(
                'python adversarial_images.py --setup'))
            sys.exit(-1)

    def attack_all(self, examples_file):
        samples = self._get_samples(examples_file)
        images = samples["images"]
        labels = samples["labels"]

        # Aggregate
        y_preds_adversarial, y_targs, noises = [], [], []
        # Attack each example 
        start = time.time()
        for x, y_true in zip(images, labels):
            # Add a channel dimension
            x = x.reshape(1, *x.shape)
            y_target = random.choice(list(set(range(10)) - set([y_true])))
            noise, y_pred, y_pred_adversarial = self.attack(x, y_true, y_target, regularization="l2")

            if y_pred == y_true:
                # store
                y_preds_adversarial.append(y_pred_adversarial)
                y_targs.append(y_target)
                noises.append(noise.squeeze())
        end = time.time()

        avg_pixel_noise = np.mean(np.abs(np.array(noises)))
        total = len(y_targs)
        success = (np.array(y_targs) == np.array(y_preds_adversarial)).sum()
        print('Attack on 5k examples took {:.3f}s'.format(end - start))
        print('Success rate {}/{} = {:.3f}'.format(success, total, 
            success / total))
        print('Average pixel perturbation {}, image perturbation {}'.format(
            avg_pixel_noise, avg_pixel_noise * 28 * 28))

    @abc.abstractmethod
    def attack(self, x, y_true, y_target):
        pass
