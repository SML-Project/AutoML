import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
#base_input=784
#base_hidden=20

#Wxh = xavier_init(size=[base_input, base_hidden])
#bxh = torch.zeros(base_hidden)

#def relu(input):
#    input[torch.where(input<0.)]=0
#    return input
#def sigmoid(input):
#    return 1/(torch.exp(input*(-1))+1)

################################################################################
#默认所有Tensor版本的运算都在cuda上进行
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class TwoLayer_Tensor():
    def __init__(self,input_units,hidden_units,activation='sigmoid'):
        self.activation = activation
        #self.input_units = input_units
        self.W_1 = self.xavier_init2(size=[input_units,hidden_units]) #初始化size为input_units*hiddenunits的W网络参数
        #print(self.W_1)
        self.b_1 = torch.zeros(hidden_units,requires_grad=True,device=device)      #初始化size为hiddenunits的b网络参
        self.W_out = self.xavier_init2(size=[hidden_units,10])   #初始化size为hidden_units*10的输出层W参数
        self.b_out = torch.zeros(10,requires_grad=True,device=device)        #初始化size为hidden_units的输出层b参数
        self.param_dict={"W_1":self.W_1,
                         "b_1":self.b_1,
                         "W_out":self.W_out,
                         "b_out":self.b_out}


    def load_params(self,params):
        if isinstance(params,dict):
            # print(params["W_1"].size())
            self.param_dict["W_1"] = params["W_1"]
            self.param_dict["b_1"] = params["b_1"]
            self.param_dict["W_out"] = params["W_out"]
            self.param_dict["b_out"] = params["b_out"]
            self.W_1 = self.param_dict["W_1"]
            self.W_out = self.param_dict["W_out"]
            self.b_1 = self.param_dict["b_1"]
            self.b_out = self.param_dict["b_out"]
        else:
            self.param_dict["W_1"] = params[0]
            self.param_dict["b_1"] = params[1]
            self.param_dict["W_out"] = params[2]
            self.param_dict["b_out"] = params[3]
            self.W_1 = self.param_dict["W_1"]
            self.W_out = self.param_dict["W_out"]
            self.b_1 = self.param_dict["b_1"]
            self.b_out = self.param_dict["b_out"]
        return

    def forward(self,X):
        #X=X.to(device)
        in_size = X.size(0)  # one batch     此时的x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值    把batchsize的值作为网络的in_size
        X = X.reshape(in_size, -1)
        #print(X.size())
        #print(X.size())
        #print("W_1size",self.W_1.size())
        if self.activation=="sigmoid":
            X=torch.sigmoid(torch.matmul(X,self.W_1)+self.b_1)
        else:
            X = torch.relu(torch.matmul(X, self.W_1) + self.b_1)
        X=torch.matmul(X,self.W_out)+self.b_out
        return X

    def xavier_init2(self,size):
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        return Variable(torch.randn(*size,device=device) * xavier_stddev, requires_grad=True)

    def xavier_init(self,size):
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        out=torch.randn(size) * xavier_stddev
        out=out.requires_grad_()
        #out.requires_grad_()
        return out
########################################################################################
class ThreeLayer_Tensor():
    def __init__(self,input_units,hidden_units,activation='sigmoid'):
        self.activation = activation
        #self.input_units = input_units
        self.W_1 = self.xavier_init2(size=[input_units,hidden_units]) #初始化size为input_units*hiddenunits的W网络参数
        #print(self.W_1)
        self.b_1 = torch.zeros(hidden_units,requires_grad=True,device=device)      #初始化size为hiddenunits的b网络参

        self.W_2 = self.xavier_init2(size=[hidden_units, hidden_units])  # 初始化size为input_units*hiddenunits的W网络参数
        # print(self.W_1)
        self.b_2 = torch.zeros(hidden_units, requires_grad=True, device=device)  # 初始化size为hiddenunits的b网络参

        self.W_out = self.xavier_init2(size=[hidden_units,10])   #初始化size为hidden_units*10的输出层W参数
        self.b_out = torch.zeros(10,requires_grad=True,device=device)        #初始化size为hidden_units的输出层b参数
        self.param_dict={"W_1":self.W_1,
                         "b_1":self.b_1,
                         "W_2": self.W_2,
                         "b_2": self.b_2,
                         "W_out":self.W_out,
                         "b_out":self.b_out}


    def load_params(self,params):
        if isinstance(params,dict):
            # print(params["W_1"].size())
            self.param_dict["W_1"] = params["W_1"]
            self.param_dict["b_1"] = params["b_1"]
            self.param_dict["W_2"] = params["W_2"]
            self.param_dict["b_2"] = params["b_2"]
            self.param_dict["W_out"] = params["W_out"]
            self.param_dict["b_out"] = params["b_out"]

            self.W_1 = self.param_dict["W_1"]
            self.W_2 = self.param_dict["W_2"]
            self.W_out = self.param_dict["W_out"]
            self.b_1 = self.param_dict["b_1"]
            self.b_2 = self.param_dict["b_2"]
            self.b_out = self.param_dict["b_out"]
        else:
            self.param_dict["W_1"] = params[0]
            self.param_dict["b_1"] = params[1]
            self.param_dict["W_2"] = params[2]
            self.param_dict["b_2"] = params[3]
            self.param_dict["W_out"] = params[4]
            self.param_dict["b_out"] = params[5]
            self.W_1 = self.param_dict["W_1"]
            self.W_2 = self.param_dict["W_2"]
            self.W_out = self.param_dict["W_out"]
            self.b_1 = self.param_dict["b_1"]
            self.b_2 = self.param_dict["b_2"]
            self.b_out = self.param_dict["b_out"]
        return

    def forward(self,X):
        #X=X.to(device)
        in_size = X.size(0)  # one batch     此时的x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值    把batchsize的值作为网络的in_size
        X = X.reshape(in_size, -1)
        #print(X.size())
        #print(X.size())
        #print("W_1size",self.W_1.size())
        if self.activation=="sigmoid":
            X=torch.sigmoid(torch.matmul(X,self.W_1)+self.b_1)
            X = torch.sigmoid(torch.matmul(X, self.W_2) + self.b_2)
        else:
            X = torch.relu(torch.matmul(X, self.W_1) + self.b_1)
            X = torch.relu(torch.matmul(X, self.W_2) + self.b_2)
        X=torch.matmul(X,self.W_out)+self.b_out
        return X

    def xavier_init2(self,size):
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        return Variable(torch.randn(*size,device=device) * xavier_stddev, requires_grad=True)
