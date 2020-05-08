import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerNet(nn.Module):
    def __init__(self,hidden_units,activation='sigmoid'):
        super(TwoLayerNet, self).__init__()
        self.activation=activation
        self.hidden_1= hidden_units
        self.fc_1 = nn.Linear(784,self.hidden_1 )  # （in_features, out_features）
        self.out=nn.Linear(self.hidden_1,10)

    def forward(self,x):
        in_size = x.size(0)  # one batch     此时的x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值    把batchsize的值作为网络的in_size
        x = x.view(in_size, -1)  # flatten the tensor 相当于resharp
        # print(x.size())
        # x: 64*320
        if self.activation=='sigmoid':
            x = torch.sigmoid(self.fc_1(x))
        else:
            x=F.relu(self.fc_1(x))
        # print(x.size())
        # x:64*20
        x = self.out(x)
        # print(x.size())
        # x:64*10

        return x # 64*10

class ThreeLayerNet(nn.Module):
    def __init__(self,hidden_units,activation='sigmoid'):
        super(ThreeLayerNet, self).__init__()
        self.hidden_1= hidden_units
        self.hidden_2 = hidden_units
        self.fc_1 = nn.Linear(784,self.hidden_1 )  # （in_features, out_features）
        self.fc_2 = nn.Linear(self.hidden_1 , self.hidden_2)  # （in_features, out_features)
        self.out=nn.Linear(self.hidden_2,10)

    def forward(self,x):
        in_size = x.size(0)  # one batch     此时的x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值    把batchsize的值作为网络的in_size
        x = x.view(in_size, -1)  # flatten the tensor 相当于resharp
        # print(x.size())
        # x: 64*320
        if self.activation=='sigmoid':
            x = torch.sigmoid(self.fc_1(x))
        else:
            x=F.relu(self.fc_1(x))
        # print(x.size())
        # x:64*20
        if self.activation=='sigmoid':
            x = torch.sigmoid(self.fc_2(x))
        else:
            x=F.relu(self.fc_2(x))
        x = self.out(x)
        # print(x.size())
        # x:64*10

        return x # 64*10

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 40, 3)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(40, 10)#（in_features, out_features）

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch     此时的x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值    把batchsize的值作为网络的in_size
        # x: 64*1*28*28
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*10*12*12  feature map =[(28-4)/2]^2=12*12
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*20*4*4
        x = F.relu(self.mp(self.conv3(x)))
        x = x.view(in_size, -1) # flatten the tensor 相当于resharp
        # print(x.size())
        # x: 64*320
        x = self.fc(x)
        # x:64*10
        # print(x.size())
        return x #64*10

class ConvCIFARNet(nn.Module):
    def __init__(self):
        super(ConvCIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)

        self.fc1 = nn.Linear(128, 32,bias=True)
        self.out = nn.Linear(32, 10,bias=True)

    def forward(self, x):
        in_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #print(x.size())
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

class TwoLayerCIFARNet(nn.Module):
    def __init__(self,hidden_units,activation='sigmoid'):
        super(TwoLayerCIFARNet, self).__init__()
        self.hidden_1= hidden_units
        self.fc_1 = nn.Linear(3*32*32,self.hidden_1 )  # （in_features, out_features）
        self.out=nn.Linear(self.hidden_1,10)

    def forward(self,x):
        in_size = x.size(0)  # one batch     此时的x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值    把batchsize的值作为网络的in_size
        x = x.view(in_size, -1)  # flatten the tensor 相当于resharp
        # print(x.size())
        # x: 64*320
        if self.activation=='sigmoid':
            x = torch.sigmoid(self.fc_1(x))
        else:
            x=F.relu(self.fc_1(x))
        # print(x.size())
        # x:64*20
        x = self.out(x)
        # print(x.size())
        # x:64*10

        return x # 64*10

class ThreeLayerCIFARNet(nn.Module):
    def __init__(self,hidden_units,activation='sigmoid'):
        super(ThreeLayerCIFARNet, self).__init__()
        self.hidden_1= hidden_units
        self.hidden_2 = hidden_units
        self.fc_1 = nn.Linear(3*32*32,self.hidden_1 )  # （in_features, out_features）
        self.fc_2 = nn.Linear(self.hidden_1 , self.hidden_2)  # （in_features, out_features)
        self.out=nn.Linear(self.hidden_2,10)

    def forward(self,x):
        in_size = x.size(0)  # one batch     此时的x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值    把batchsize的值作为网络的in_size
        x = x.view(in_size, -1)  # flatten the tensor 相当于resharp
        # print(x.size())
        # x: 64*320
        x = torch.sigmoid(self.fc_1(x))
        # print(x.size())
        # x:64*20
        x = torch.sigmoid(self.fc_2(x))
        x = self.out(x)
        # print(x.size())
        # x:64*10

        return x # 64*10