import argparse
import sys

import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import set_param
from LoadData import get_data_loader
import Models


parser = argparse.ArgumentParser(description='Parameters for learning to learn')

parser.add_argument('--USE_CUDA', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--LSTM_TRAIN_ITERATION', type=int, default=100, metavar='N',
                    help='# of meta optimizer steps (default: 100)')
parser.add_argument('--UNROLL_ITERATION', type=int, default=20, metavar='N',
                    help='# of iterations for loss function (default: 20)')
parser.add_argument('--THETA_RESET_INTERVAL', type=int, default=10, metavar='N',
                    help='# of iterations to reset parameters of optimizee (default: 10)')
parser.add_argument('--LSTM_ADAM_LR', type=float, default=0.001, metavar='N',
                    help='learning rate for ADAM (default: 0.001)')
parser.add_argument('--LSTM_ADAM_BETAS', type=float, default=0.9, metavar='N',
                    help='coefficient for averaging gradient (default: 0.9)')
parser.add_argument('--LSTM_ADAM_EPS', type=float, default=1e-8, metavar='N',
                    help='additional param for stability (default: 1e-8)')
parser.add_argument('--LSTM_ADAM_WD', type=int, default=0, metavar='N',
                    help='weight_decay (default: 0)')
parser.add_argument('--LR', type=float, default=0.001, metavar='N',
                    help='learning rate (default: 0.001)')

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--num_epochs', type=int, default=20000, metavar='N',
                    help='# of epochs (default: 20000)')
parser.add_argument('--num_stack', type=int, default=2, metavar='N',
                    help='# of LSTM layers (default: 2)')
parser.add_argument('--preprocess', action='store_true', default=True,
                    help='enables LSTM preprocess')
parser.add_argument('--p', type=int, default=10, metavar='N',
                    help='criterion for preprocess of gradient (default: 10)')
parser.add_argument('--output_scale', type=float, default=0.1, metavar='N',
                    help='scale for updates of gradient (default: 0.1)')

args = parser.parse_args()
args.USE_CUDA = args.USE_CUDA and torch.cuda.is_available()

def w(v):
    if torch.cuda.is_available():
        return v.cuda()
    return v
class QuadraticLoss:
    def __init__(self, **kwargs):
        self.W = w(Variable(torch.randn(10, 10)))
        self.y = w(Variable(torch.randn(10).t()))

    def get_loss(self, theta):
        return torch.sum((self.W.matmul(theta) - self.y) ** 2)
loss_func = QuadraticLoss()
loss_func.W.requires_grad = True
loss_func.y.requires_grad = True

x = torch.randn(10).t()
x.requires_grad = True

opt_SGD = optim.SGD([x], lr=1e-3)
opt_Momentum = optim.SGD([x], lr=args.LR, momentum=0.8)
opt_RMSprop = optim.RMSprop([x], lr=args.LR, alpha=0.9)
opt_Adam = optim.Adam([x], lr=args.LR, betas=(0.9,0.99))
opt_dict = {"opt_SGD": opt_SGD,"opt_Momentum": opt_Momentum,"opt_RMSprop": opt_RMSprop,"opt_Adam": opt_Adam}


def train_quadratic(optimizer):
    for step in range(args.num_epochs):
        #pred = torch.sum((loss_func.W.matmul(x) - loss_func.y) ** 2)
        pred = loss_func.get_loss(x)
        optimizer.zero_grad()
        pred.backward()
        optimizer.step()
        if step % 2000 == 0:
            print('step{}: x = {}, f(x) = {}'.format(step, x, pred.item()))

def train_opts_quadratic(**kwargs):
    for key, value in kwargs.items():
        optimizer = kwargs[key]
        print(key)
        train_quadratic(optimizer)

'''
def train_MNIST(net,optimizer):
    #net = Models.ConvNet()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.num_epochs):
        train_loss = []
        for batch_id, (data, target) in enumerate(get_data_loader()):
            print(batch_id)
            net.train()
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss)
            if batch_id % 100 == 0:
                net.eval()
                val_loss = []
                for (data, target) in get_data_loader(train=False):
                    output = net(data)
                    loss_val = criterion(output, target)
                    val_loss.append(loss_val)
                print(val_loss)
    test_loss = []
    with torch.no_grad():
        for data, target in get_data_loader(train=False):
            output = net(data)
            loss_test = criterion(output, target)
            test_loss.append(loss_test)
    print(val_loss)

def train_opts_MNIST(nettype="ConvNet",**kwargs):
    for key, value in kwargs.items():
        optimizer = kwargs[key]
        print(key)
        if nettype == "TwoLayerNet":
            net = Models.TwoLayerNet()
        elif nettype == "ThreeLayerNet":
            net = Models.ThreeLayerNet()
        elif nettype == "ConvNet":
            net = Models.ConvNet()

        train_MNIST(net,optimizer)
'''


def train_MNIST_CIFAR(net, dataset_name, optimizer):
    #net = Models.ConvNet()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.num_epochs):
        train_loss = []
        for batch_id, (data, target) in enumerate(get_data_loader(dataset_name = dataset_name)):
            #print(batch_id)
            net.train()
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss)
            if batch_id % 100 == 0:
                net.eval()
                val_loss = []
                for (data, target) in get_data_loader(dataset_name = dataset_name, train=False):
                    output = net(data)
                    loss_val = criterion(output, target)
                    val_loss.append(loss_val.item())
                #print(val_loss)
    test_loss = []
    with torch.no_grad():
        for data, target in get_data_loader(dataset_name = dataset_name, train=False):
            output = net(data)
            loss_test = criterion(output, target)
            test_loss.append(loss_test.item())
    return val_loss, test_loss

def train_opts_MNIST_CIFAR(net_type="ConvNet", dataset_name = 'MNIST'):

    if net_type == "TwoLayerNet":
        net = Models.TwoLayerNet()
    elif net_type == "ThreeLayerNet":
        net = Models.ThreeLayerNet()
    elif net_type == "ConvNet":
        net = Models.ConvNet()
    elif net_type == "ConvCIFARNet":
        net = Models.ConvCIFARNet()
    elif net_type == "TwoLayerCIFARNet":
        net = Models.TwoLayerCIFARNet()
    elif net_type == "ThreeLayerCIFARNet":
        net = Models.ThreeLayerCIFARNet()

    opt_SGD = optim.SGD(net.parameters(), lr=1e-3)
    opt_Momentum = optim.SGD(net.parameters(), lr=args.LR, momentum=0.8)
    opt_RMSprop = optim.RMSprop(net.parameters(), lr=args.LR, alpha=0.9)
    opt_Adam = optim.Adam(net.parameters(), lr=args.LR, betas=(0.9, 0.99))
    opt_dict_set = {"opt_SGD": opt_SGD, "opt_Momentum": opt_Momentum, "opt_RMSprop": opt_RMSprop, "opt_Adam": opt_Adam}

    for key, value in opt_dict_set.items():
        optimizer = opt_dict_set[key]
        print(key)
        val_losslist, test_losslist = train_MNIST_CIFAR(net, dataset_name, optimizer)
        filename = net_type + dataset_name + key + '.txt'
        with open(filename, 'a') as f_object:
            f_object.write("val_losslist" + '\n' + str(val_losslist))
            f_object.write("test_losslist" + '\n' + str(test_losslist))


if __name__ == '__main__':
    #train_opts_quadratic(**opt_dict)
    train_opts_MNIST_CIFAR()

