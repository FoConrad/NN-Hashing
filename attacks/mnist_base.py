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

CNN_MODEL_WEIGHTS_FILE = 'cnn_model_weights.pkl'

# Adversarial example MNIST image generation code borrowed and inspired from:
# https://github.com/akshaychawla/Adversarial-Examples-in-PyTorch

class MNISTAttack(object):
    def __init__(self, model_class, model_file):
        self._model_class = model_class
        self._model = self.load_target_model()

    @staticmethod
    def load_taget_model():
        model = self._model_class(data_err=True)
        weights = {}
        try:
            with open(CNN_MODEL_WEIGHTS_FILE, 'rb') as f:
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


def train_target(model_class, trainloader, epochs=5, testloader=None,
        outfile=None):
    model = model_class(data_err=False).cudize()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=.01, momentum=0.5)
    # TRAIN 
    for epoch in range(epochs):
        running_loss = 0.0 
        for data in trainloader:
            inputs, labels = data 
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad() 
            outputs = model(inputs) # forward pass 
            l = loss(outputs, labels) # compute softmax -> loss 
            l.backward() # get gradients on params 
            optimizer.step() # SGD update 
            # print statistics 
            running_loss += l.data[0]
        print("Epoch {}/{} | Loss: {}".format(epoch + 1, epochs,
            running_loss/2000), end="\r")
    print()
    if testloader is None: return

    correct = 0.0 
    total = 0 
    for data in testloader:
        images, labels = data 
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) 
        correct += (predicted == labels).sum()
    print("Accuracy: {}".format(correct/total))
    
    if outfile is None: return
    print('Writing to file', outfile, end='')
    weights_dict = {} 
    for param in list(model.named_parameters()):
        weights_dict[param[0]] = param[1] 
    with open(outfile,"wb") as f:
        pickle.dump(weights_dict, f)
    print('...done')

         

if __name__ == '__main__':
    from functools import partial
    import torchvision
    from torchvision import transforms
    from models import CNN

    mnist_transform = transforms.ToTensor()
    traindata = torchvision.datasets.MNIST(root="./mnist", train=True, download=True, transform=mnist_transform)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers=2)

    testdata  = torchvision.datasets.MNIST(root="./mnist", train=False, download=True, transform=mnist_transform)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=True, num_workers=2)


    _cudize = lambda in_: in_.cuda() if False else in_
    m = partial(CNN, (28,28), 6, 10, _cudize)

    train_target(m, trainloader, epochs=15, testloader=testloader,
            outfile=CNN_MODEL_WEIGHTS_FILE)
