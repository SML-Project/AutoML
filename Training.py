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
from LSTMLearner import LSTMLearner

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

args = parser.parse_args()
args.USE_CUDA = args.USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda:0" if args.USE_CUDA else "cpu")


def train_mnist_cifar_base(net, dataset_name, criterion, optimizer, args=args):
    t_loss = []
    for epoch in range(args.num_epochs):
        train_loss = 0
        train_loader = get_data_loader(dataset_name=dataset_name)
        for batch_id, (data, target) in enumerate(train_loader):
            #print(batch_id, data.size(), target.size(), data.size()[1] * data.size()[2],len(train_loader.dataset), len(train_loader))
            data, target = data.to(device), target.to(device)
            net.train()
            optimizer.zero_grad()
            output = net(data)
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
    learner_b = LSTMLearner(args)
    train_loader = get_data_loader(dataset_name=dataset_name)
    learner_a.learn(net, criterion,train_loader)
    learner_b.learn(net, criterion,train_loader)
    learner_a.init_step(net.convlayer)
    learner_a.init_step(net.fclayer)
    t_loss = []
    for epoch in range(args.num_epochs):
        train_loss = 0
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            net.train()
            output = net(data)
            optimizer_conv = learner_a(net.convlayer.parameters())
            optimizer_fc = learner_b(net.fclayer.parameters())
            optimizer_conv.zero_grad(net.convlayer)
            optimizer_fc.zero_grad(net.fclayer)
            loss = criterion(output, target)
            loss.backward()
            optimizer_conv.step(net.convlayer)
            optimizer_fc.step(net.fclayer)
            train_loss += loss.item()
            if batch_id % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_id * len(data), len(train_loader.dataset),
                           100. * batch_id / len(train_loader), loss.item()))
        train_loss /= len(train_loader)
        t_loss.append(train_loss)
    return t_loss


'''
def test_MNIST_CIFAR_base(args = args, net, dataset_name, optimizer):
    net.eval()
    loss_test = 0
    test_loss = []
    with torch.no_grad():
        test_loader = get_data_loader(dataset_name = dataset_name, train=False)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss_test += criterion(output, target, size_average = False).item()
            #test_loss.append(loss_test.item())
    loss_test /= len(test_loader.dataset)
    return loss_test
'''


def get_data_size(dataset_name="Quadratic_Origin"):
    for batch_id, (data, target) in enumerate(get_data_loader(dataset_name)):
        input_units = data.size()[1] * data.size()[2]
    return input_units


def train_opts_mnist_cifar(net_type, dataset_name, criterion, optimizer, args=args):
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


if __name__ == '__main__':
    # Quadratic function
    train_opts_mnist_cifar(net_type="Qua", dataset_name="Quadratic_Origin", criterion=nn.MSELoss(), optimizer="opt_SGD")

    #train_opts_mnist_cifar(net_type="ConvMNISTNet", dataset_name='MNIST', criterion = nn.CrossEntropyLoss(), optimizer = "LSTMLearner")

'''
train_opts_mnist_cifar()
Choose the type for net:
    net_type="Qua"
    net_type="TwoLayerNet"
    net_type="ThreeLayerNet"
    net_type="ConvMNISTNet"
    net_type="ConvCIFARNet"
Choose the name for dataset:
    dataset_name='MNIST'
    dataset_name='CIFAR10'
    dataset_name='CIFAR2'
    dataset_name='CIFAR5'
    dataset_name='Quadratic_Origin'
    dataset_name='Quadratic_Uniform'
    dataset_name='Quadratic_Gauss'
Choose the type for criterion:
    criterion=nn.MSELoss() 
    criterion=nn.CrossEntropyLoss()
    criterion=nn.NLLLoss() 
    ...
Choose the type for optimizer:
   optimizer="LSTMLearner" 
   optimizer="opt_SGD"
   optimizer="opt_Momentum"
   optimizer="opt_RMSprop"
   optimizer="opt_Adam"
'''