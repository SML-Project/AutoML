import Models
from NewTrain import train_mnist_cifar_learner,args,get_net_params,train_mnist_cifar_base
import torch.nn as nn
from LSTMLearnerTensor import LSTMLearner
from LoadData import get_data_loader
import torch.optim as optim

net_type="TwoLayer_Tensor"
dataset_name="MNIST"
criterion_name="CrossEntropy"
optimizer_name="LSTMLearner"

net=Models.TwoLayer_Tensor(input_units=784,hidden_units=20)
params=get_net_params(net)


#optimizer=optim.Adam(params, lr=args.LR)
criterion=nn.CrossEntropyLoss()

#loss_his=train_mnist_cifar_base(net, "MNIST", nn.CrossEntropyLoss(), optimizer, args=args)
#loss_his=train_mnist_cifar_learner(net, params, dataset_name, criterion, args=args)
learner_a = LSTMLearner(args)
train_loader = get_data_loader(dataset_name=dataset_name)
learner_a.learn(params,net,criterion,train_loader)

#print(len(loss_his))
'''
path = "model_parameters/modelparam_1.pth"
torch.save(net.param_dict,path)
filename = net_type + dataset_name + criterion_name + optimizer_name + '.txt'
torch.save(loss_his,filename)
'''