import argparse
import pickle
import os
import sys
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import transforms

from models import CNN

# Adversarial example MNIST image generation code borrowed and inspired from:
# https://github.com/akshaychawla/Adversarial-Examples-in-PyTorch

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


def setup_model(outfile):
    mnist_transform = transforms.ToTensor()
    traindata = torchvision.datasets.MNIST(root="./mnist", train=True, download=True, transform=mnist_transform)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers=2)
    testdata  = torchvision.datasets.MNIST(root="./mnist", train=False, download=True, transform=mnist_transform)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=True, num_workers=2)

    _cudize = lambda in_: in_
    m = partial(CNN, (28,28), 6, 10, _cudize)
    train_target(m, trainloader, epochs=15, testloader=testloader,
            outfile=outfile)

def setup_5k_examples(example_file):
    mnist_transform = transforms.ToTensor()
    testdata  = torchvision.datasets.MNIST(root="./mnist", train=False, download=True, transform=mnist_transform)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=100, shuffle=True, num_workers=1)

    images, labels = [], []
    for idx, data in enumerate(testloader):
        x_lots, y_lots = data 
        for x,y in zip(x_lots, y_lots):
            images.append(x.numpy())
            labels.append(y)
        if idx==49:
            break

    print('Writing examples file', example_file, end='')
    with open(example_file, "wb") as f: 
        data_dict = { "images":images, "labels": labels}
        pickle.dump(data_dict, f)
    print('...done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack CNN MNIST.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--setup', action='store_true',
            help='train target model and generate 5k examples for testing')
    args = parser.parse_args()

    base_dir = 'image_misc'
    cnn_model_weights = os.path.join(base_dir, 'cnn_model_weights.pkl')
    mnist_5k_examples = os.path.join(base_dir, '5k_samples.pkl')

    if args.setup:
        if not os.path.isdir(base_dir):
            os.mkdir(base_dir)
        setup_model(cnn_model_weights)
        setup_5k_examples(mnist_5k_examples)
        print('Finished setting up... exiting')
        sys.exit(0)

