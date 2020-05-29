import argparse

import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from LoadData import get_data_loader
import Models
from LSTMLearner_tensor import LSTMLearner
import matplotlib.pyplot as plt

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
parser.add_argument('--hidden_size', type=int, default=20, metavar='N',
                    help='dim of LSTM hidden states (default: 20)')

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--num_epochs', type=int, default=20000, metavar='N',
                    help='# of epochs (default: 20000)')
parser.add_argument('--num_stacks', type=int, default=2, metavar='N',
                    help='# of LSTM layers (default: 2)')
parser.add_argument('--preprocess', action='store_true', default=True,
                    help='enables LSTM preprocess')
parser.add_argument('--p', type=int, default=10, metavar='N',
                    help='criterion for preprocess of gradient (default: 10)')
parser.add_argument('--output_scale', type=float, default=0.1, metavar='N',
                    help='scale for updates of gradient (default: 0.1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_known_args()[0]
args.USE_CUDA = args.USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda:0" if args.USE_CUDA else "cpu")

def train_mnist_cifar_base(net, dataset_name, criterion, optimizer, args=args):
    t_loss = []
    #net=net.to(device)
    #for epoch in range(args.num_epochs):
    for epoch in range(10):
        train_loss = 0
        train_loader = get_data_loader(dataset_name=dataset_name)
        for batch_id, (data, target) in enumerate(train_loader):
            #print(batch_id, data.size(), target.size(), data.size()[1] * data.size()[2],len(train_loader.dataset), len(train_loader))
            data, target = data.to(device), target.to(device)
            #net.train()
            optimizer.zero_grad()
            output = net.forward(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_id % args.log_interval == 0:
                print(str(batch_id))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_id * len(data), len(train_loader.dataset),
                           100. * batch_id / len(train_loader), loss.item()))
        train_loss /= len(train_loader)
        t_loss.append(train_loss)
    return t_loss

def train_mnist_cifar_learner(net, dataset_name, criterion, args=args):
    learner_a = LSTMLearner(args)
    #learner_b = LSTMLearner(args)
    train_loader = get_data_loader(dataset_name=dataset_name)
    #learner_a.learn(params,net, criterion,train_loader)
    #learner_b.learn(net, criterion,train_loader)
    learner_a.init_step(net.parameters())
    #learner_a.init_step(net.fclayer)
    t_loss = []
    for epoch in range(5):
        train_loss = 0
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            #net.train()
            output = net(data)
            #optimizer = learner_a(params)
            #optimizer_fc = learner_b(net.fclayer.parameters())
            #optimizer.zero_grad(params)
            learner_a.zero_grad(net.parameters())
            #optimizer_fc.zero_grad(net.fclayer)
            loss = criterion(output, target)
            loss.backward(retain_graph=True)
            learner_a.step(net)
            #optimizer_fc.step(net.fclayer)
            train_loss += loss.item()
            if batch_id % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_id * len(data), len(train_loader.dataset),
                           100. * batch_id / len(train_loader), loss.item()))
        train_loss /= len(train_loader)
        t_loss.append(train_loss)
        torch.save(net.state_dict(),"fc2_model_sigmoid.pkl")
    return t_loss


def train_opts_mnist_cifar(net_type, dataset_name, criterion, optimizer, args=args,params=None):
    if net_type == "Qua":
        input_units = get_data_size()
        net = Models.Qua(input_units).to(device)
    elif net_type == "TwoLayerNet":
        net = Models.TwoLayerNet().to(device)
    elif net_type == "ThreeLayerNet":
        net = Models.ThreeLayerNet().to(device)
    elif net_type == "ConvMNISTNet":
        net = Models.ConvMNISTNet().to(device)
    elif net_type == "ConvCIFARNet":
        net = Models.ConvCIFARNet().to(device)

    opt_SGD = optim.SGD(net.parameters(), lr=args.LR)
    opt_Momentum = optim.SGD(net.parameters(), lr=args.LR, momentum=0.8)
    opt_RMSprop = optim.RMSprop(net.parameters(), lr=args.LR, alpha=0.9)
    opt_Adam = optim.Adam(net.parameters(), lr=args.LR, betas=(0.9, 0.99))
    opt_dict_set = {"opt_SGD": opt_SGD, "opt_Momentum": opt_Momentum, "opt_RMSprop": opt_RMSprop, "opt_Adam": opt_Adam}

    if optimizer == "LSTMLearner":
        t_loss = train_mnist_cifar_learner(net, dataset_name, criterion, args=args)

    for key, value in opt_dict_set.items():
        if optimizer == key:
            print(key)
            optimizer = opt_dict_set[key]
            t_loss = train_mnist_cifar_base(net, dataset_name, criterion, optimizer, args=args)
            filename = net_type + dataset_name + criterion + key +'.txt'
            with open(filename, 'a') as f_object:
                for epoch, loss in enumerate(t_loss):
                    f_object.write("Epoch: " + str(epoch) + ' ' + "Loss: " + str(loss))

def get_net_params(net):
    params = []
    for key, value in net.param_dict.items():
        params.append(value)
    return params


