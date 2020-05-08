import torch
import numpy as np
from torchvision.datasets import mnist,CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.utils.data as Data

'''
a general method to load various type of data from external files
'''
'''
def LoadData(N, batch_size, num_workers):
    X = torch.unsqueeze(torch.rand(N) * 4 - 2, dim=1)
    Y = X * (8 * X).sin() * (4 / X).sin()
    X_train, X_valid = X.split([int(0.7*N), int(0.3*N)], dim=0)
    Y_train, Y_valid = Y.split([int(0.7*N), int(0.3*N)], dim=0)
    data_train = TensorDataset(X_train, Y_train)
    data_valid = TensorDataset(X_valid, Y_valid)
    data = {"train": data_train, "valid": data_valid}
    for phase in ["train", "valid"]:
        print(phase + "_data: " + str(len(data[phase])))
    print("\n")
    loaders = {phase: DataLoader(data[phase], batch_size=batch_size, shuffle=True,  num_workers=num_workers)
                for phase in ["train", "valid"]}
    return loaders
'''
def get_data_loader(dataset_name='MNIST',batch_size=32,train=True,transform=transforms.ToTensor()):
    dataset=None
    if dataset_name=='MNIST':
        if train==True:
            dataset = mnist.MNIST('./mnist_dataset', train=True, download=False, transform=transform)
        else:
            dataset = mnist.MNIST('./mnist_dataset', train=False, download=False, transform=transform)
    elif dataset_name == 'CIFAR10':
        if train==True:
            dataset = CIFAR10(root='./cifar10_dataset', train=True, download=False, transform=transform)
        else:
            dataset = CIFAR10(root='./cifar10_dataset', train=False, download=False, transform=transform)
    elif dataset_name=='CIFAR2':
        if train == True:
            data,label=torch.load('./cifar10_dataset/data/CIFAR2.pt')
            dataset = Data.TensorDataset(data, label)
        else:
            data, label = torch.load('./cifar10_dataset/data/CIFAR2_test.pt')
            dataset = Data.TensorDataset(data, label)
    elif dataset_name=='CIFAR5':
        if train == True:
            data,label=torch.load('./cifar10_dataset/data/CIFAR5.pt')
            dataset = Data.TensorDataset(data, label)
        else:
            data, label = torch.load('./cifar10_dataset/data/CIFAR5_test.pt')
            dataset = Data.TensorDataset(data, label)
    if dataset==None:
        raise ('dataset is None')
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader