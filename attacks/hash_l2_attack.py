import itertools

import torch
import torch.nn as nn
from torch import optim

from . import HashBaseAttack

class HashL2Attack(HashBaseAttack):
    def __init__(self, adam_lr=1., reg_coeff=0.0045, **kwargs):
        super().__init__(**kwargs)
        self._reg_coeff = reg_coeff
        self._loss = nn.MSELoss()
        self._optimizer = optim.Adam(params=[self._model.r], lr=adam_lr)


    # W function with zeros as 0 and 1, the two valid binary inputs
    def loss_reg(self, r):
        x = self._source['param'] + r
        ret = torch.mean(torch.pow(x * (x - 1), 2))
        return ret * self._reg_coeff

    def iterate(self, steps=0):
        self.report_hash(self._source['output'], 'original source hash')
        self.report_hash(self._target['output'], 'target hash')

        success = False
        for iteration in range(steps) if steps else itertools.count():
            self._optimizer.zero_grad()
            outputs = self._model(self._source['data'], ignore_data_err=False)
            loss = self._loss(outputs, self._target['output'])
            adv_loss = loss + self.loss_reg(self._model.r)
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
            self._optimizer.step()

        print('\n...{}'.format('success' if success else 'failure'))
