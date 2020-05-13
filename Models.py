import torch
import torch.nn as nn
import torch.nn.functional as F

class Qua(nn.Module):
    def __init__(self,input_units):
        super(Qua, self).__init__()
        self.input_units=input_units
        self.out=nn.Linear(self.input_units,10)

    def forward(self,x):

        in_size = x.size(0)  # one batch     此时的x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值    把batchsize的值作为网络的in_size
        x = x.view(in_size, -1)  # flatten the tensor 相当于reshape
        x = self.out(x)
        # print(x.size())
        # x:64*10
        return x # 64*10

class TwoLayerNet(nn.Module):
    def __init__(self,input_units,hidden_units,activation='sigmoid'):
        super(TwoLayerNet, self).__init__()
        self.activation=activation
        self.hidden_1= hidden_units
        self.input_units=input_units
        self.fc_1 = nn.Linear(self.input_units,self.hidden_1 )  # （in_features, out_features）
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
    def __init__(self,input_units,hidden_units,activation='sigmoid'):
        super(ThreeLayerNet, self).__init__()
        self.input_units=input_units
        self.hidden_1= hidden_units
        self.hidden_2 = hidden_units
        self.fc_1 = nn.Linear(input_units,self.hidden_1 )  # （in_features, out_features）
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

class ConvMNISTNet(nn.Module):
    def __init__(self):
        super(ConvMNISTNet, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.convlayer=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 20, 5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 40, 3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # fully connect
        self.fclayer = nn.Linear(40, 10)#（in_features, out_features）

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch     此时的x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值    把batchsize的值作为网络的in_size
        # x: 64*1*28*28
        x = self.convlayer(x)
        x = x.view(in_size, -1) # flatten the tensor 相当于resharp
        # print(x.size())
        # x: 64*320
        x = self.fclayer(x)
        # x:64*10
        # print(x.size())
        return x #64*10

class ConvCIFARNet(nn.Module):
    def __init__(self):
        super(ConvCIFARNet, self).__init__()
        self.convlayer=nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )

        self.fclayer=nn.Sequential(
            nn.Linear(128, 32, bias=True),
            nn.Linear(32, 10, bias=True)
        )
 

    def forward(self, x):
        in_size = x.size(0)
        #print(x.size())
        x=self.convlayer(x)
        x = x.view(in_size, -1)
        x=self.fclayer(x)
        return x