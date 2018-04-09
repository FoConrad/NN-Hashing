import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from . import MNISTAttack

class MNIST_FGSM(MNISTAttack):
    def __init__(self, model_class, weights_file):
        super().__init__(model_class, weights_file,fgsm=True)

    def attack(self, x, y_true, y_target):
        _x = Variable(torch.FloatTensor(x))
        _y_target = Variable(torch.LongTensor([y_target]))
        #print(y_true,y_target)

        self._model.r.data = _x.data
        # Classification before modification 
        y_pred =  np.argmax(self._model(_x).data.numpy())

        self._optimizer.zero_grad() 

        outputs = self._model(_x)
        xent_loss = self._loss_fn(outputs, _y_target) 
        xent_loss.backward()
        #self._optimizer.step()
        #print(self._model.r.grad)

        epsilon = 0.5
        x_grad = torch.sign(self._model.r.grad)

        x_adversarial = torch.clamp(_x - epsilon * x_grad,0,1)

        '''
        print(_x.data)
        print(x_adversarial.data)

        print('--------------------------------------')
        print(x_adversarial.data-_x.data)
        print('--------------------------------------')
        '''
        self._model.r.data = x_adversarial.data
        #print('x_adv"s type is ',type(x_adversarial))
        y_pred_adversarial = np.argmax(self._model(x_adversarial).data.numpy())

        '''
        print(y_pred)
        print(y_pred_adversarial)
        '''

        return (x_adversarial.data-_x.data).numpy(), y_pred, y_pred_adversarial 

'''
Attack on 5k examples took 6.137s
Success rate 2963/4856 = 0.610
Average pixel perturbation 0.2655390501022339, image perturbation 208.18261528015137
'''