import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from . import MNISTAttack

class MNIST_LBFGS(MNISTAttack):
    def __init__(self, model_class, weights_file, regularization="l2"):
        super().__init__(model_class, weights_file)
        assert regularization in ['l2', 'l1', None], \
                'Please choose valid regularization'
        self._regularization = regularization

    def attack(self, x, y_true, y_target, regularization=None):
        _x = Variable(torch.FloatTensor(x))
        _y_target = Variable(torch.LongTensor([y_target]))

        # Reset value of r 
        self._model.r.data = torch.zeros(28, 28)

        # Classification before modification 
        y_pred =  np.argmax(self._model(_x).data.numpy())

        # Optimization Loop 
        for iteration in range(1000):

            self._optimizer.zero_grad() 
            outputs = self._model(_x)
            xent_loss = self._loss_fn(outputs, _y_target) 

            if self._regularization == "l1":
                adv_loss = xent_loss + torch.mean(torch.abs(self._model.r))
            elif self._regularization == "l2":
                adv_loss  = xent_loss + torch.mean(torch.pow(self._model.r,2))
            else: 
                adv_loss = xent_loss

            adv_loss.backward() 
            self._optimizer.step() 

            # keep optimizing Until classif_op == _y_target
            y_pred_adversarial = np.argmax(self._model(_x).data.numpy())
            if y_pred_adversarial == y_target:
                break 

        return self._model.r.data.numpy(), y_pred, y_pred_adversarial 
