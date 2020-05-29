import torch
import os
import Models_tensor
from NewTrain import args,get_net_params,train_mnist_cifar_learner
import torch.nn as nn
from LSTMLearnerTensor import LSTMLearner
from LoadData import get_data_loader
import torch.optim as optim
import Models

net_type="TwoLayer_Tensor"
dataset_name="MNIST"
criterion_name="CrossEntropy"
optimizer_name="LSTMLearner"

net=Models.TwoLayerNet(input_units=784,hidden_units=20)
net=net.cuda()
#params=get_net_params(net)

#optimizer=optim.Adam(params, lr=args.LR)
criterion=nn.CrossEntropyLoss()

#loss_his=train_mnist_cifar_base(net, "MNIST", nn.CrossEntropyLoss(), optimizer, args=args)
loss_his=train_mnist_cifar_learner(net, dataset_name, criterion, args=args)
torch.save(net.state_dict(),"fc2_model_sigmoid.pkl")
filename = net_type + dataset_name + criterion_name + optimizer_name + '.txt'
torch.save(loss_his,filename)
