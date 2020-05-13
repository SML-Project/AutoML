import torch
import numpy as np

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
    elif dataset_name=='Quadratic_Origin':     #与论文中一致，X为10*10 iid 正态分布采样，Y为iid正态分布采样的10维向量  Training set shape: X:(40000,10,10)  Y:(4000,10)   Test set shape: X:(10000,10,10)  Y(10000,10)   
        if train==True:
            data, label = torch.load('./quadratic_dataset/origin/training.pt')
            dataset = Data.TensorDataset(data, label)
        else:
            data, label = torch.load('./quadratic_dataset/origin/test.pt')
            dataset = Data.TensorDataset(data, label)
    elif dataset_name=='Quadratic_Uniform':  #X为均匀分布随机采样的10维向量，Y=5*X**2.  Training set shape:(40000,10) Test set shape:(10000,10)
        if train==True:
            data, label = torch.load('./quadratic_dataset/uniform/training.pt')
            dataset = Data.TensorDataset(data, label)
        else:
            data, label = torch.load('./quadratic_dataset/uniform/test.pt')
            dataset = Data.TensorDataset(data, label)
    elif dataset_name=='Quadratic_Gauss':     #X为正态分布随机采样的10维向量，Y=5*X**2.  Training set shape:(40000,10) Test set shape:(10000,10)
        if train==True:
            data, label = torch.load('./quadratic_dataset/gauss/training.pt')
            dataset = Data.TensorDataset(data, label)
        else:
            data, label = torch.load('./quadratic_dataset/gauss/test.pt')
            dataset = Data.TensorDataset(data, label)
    if dataset==None:
        raise ('dataset is None')
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader