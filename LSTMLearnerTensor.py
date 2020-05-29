import torch
import copy
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import reduce

'''
LSTMModel这个类建立了一个LSTM模型，用来接收被优化模型参数传来的梯度，计算参数更新值。
大家不用修改这一部分，我只是介绍一下这个类的输入和输出。
被优化模型的参数全部展开成一维再拼接起来，存为grad_loader，维度是[total_size, 1]
hidden state是一个list，[state_h, state_c]，存为state_loader，两个元素的维度都是[1, total_size, hidden_size]
循环地取grad_loader中的一个batch，存为grad_batch，维度是[batch_size, 1]
再取state_loader中的一个batch，存为state_batch，每个元素的维度是[1, batch_size, hidden_size]
输入便是grad_batch, state_batch，输出是update_batch, new_state_bacth
将update_batch拼接成update_loader，维度与grad_loader相同，也是[total_size, 1]
用new_state_batch更新state_loader中相同位置的数据
这就是LSTMLearner.learn()的UNROLL_ITERATION中，每个epoch做的事情
'''

class LSTMModel(nn.Module):
    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_stacks = args.num_stacks
        self.preprocess = args.preprocess
        self.input_size = 2 if self.preprocess == True else 1
        self.output_size = 1
        self.p = args.p
        self.output_scale = args.output_scale
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_stacks)
        self.Linear = nn.Linear(self.hidden_size, self.output_size)
        self.USE_CUDA = args.USE_CUDA
        
    def gradient_preprocess(self, gradient):
        log = torch.log(torch.abs(gradient))
        if self.USE_CUDA:
            log = log.cuda()
        clamp_log = torch.clamp(log/self.p, min=-1.0, max=1.0)
        exp_p = torch.exp(torch.Tensor([self.p]))
        if self.USE_CUDA:
            exp_p = exp_p.cuda()
        clamp_sign = torch.clamp(exp_p * gradient, min=-1.0, max=1.0)
        gradient = torch.cat((clamp_log, clamp_sign), dim=-1)
        return gradient
    
    def gradient_update(self, gradient, prev_state):
        update, cur_state = self.lstm(gradient, prev_state)
        update = self.Linear(update) * self.output_scale 
        return update, cur_state
    
    def forward(self, gradient, prev_state):
        gradient = gradient.unsqueeze(0)
        if self.preprocess == True:
            gradient = self.gradient_preprocess(gradient)
        update, cur_state = self.gradient_update(gradient, prev_state)
        update = update.squeeze(0)
        return update, cur_state


'''
learn()可能需要根据model的具体形式修改，优化LSTM要用到的函数基本都不用改
init_step()和step()是learn()完成后，用来在新model上检验的，这部分不确定写好，等learn()确认无误后再改吧
'''
class LSTMLearner(object):
    def __init__(self, args):
        self.lstm_model = LSTMModel(args)
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.num_stacks = args.num_stacks
        self.USE_CUDA = args.USE_CUDA
        self.LSTM_TRAIN_ITERATION = args.LSTM_TRAIN_ITERATION
        self.UNROLL_ITERATION = args.UNROLL_ITERATION
        self.THETA_RESET_INTERVAL = args.THETA_RESET_INTERVAL
        self.LSTM_ADAM_LR = args.LSTM_ADAM_LR
        self.LSTM_ADAM_BETAS = args.LSTM_ADAM_BETAS
        self.LSTM_ADAM_EPS = args.LSTM_ADAM_EPS
        self.LSTM_ADAM_WD = args.LSTM_ADAM_WD
        self.LSTM_ADAM = torch.optim.Adam(self.lstm_model.parameters(), lr=self.LSTM_ADAM_LR)
                                          #betas=self.LSTM_ADAM_BETAS, eps=self.LSTM_ADAM_EPS,)
                                          #weight_decay=self.LSTM_ADAM_WD)
        self.LSTM_TRAIN_LOSS = 0
        self.LSTM_TRAIN_LOSS_HIST = []
        if self.USE_CUDA:
            self.lstm_model = self.lstm_model.cuda()

    def zero_grad(self, params):
        for p in params:
            
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def weight_reset(self, m):
        if isinstance(m, nn.Module):
            
            m.reset_parameters()
    
    '''
    每次要learn一个model的参数model_theta_list，都要先把LSTMModel中的参数重新初始化
    '''
    def lstm_reset(self):
        self.LSTM_TRAIN_LOSS = 0
        self.LSTM_TRAIN_LOSS_HIST = []
        self.lstm_model.lstm.apply(self.weight_reset)
        self.lstm_model.Linear.apply(self.weight_reset)

    '''
    当LSTM的迭代经过THETA_RESET_INTERVAL步后，为了防止LSTM对固定的theta_list过拟合，应该对theta_list重新初始化
    '''
    def theta_reset(self, theta_list):
        new_theta_list = []
        for theta in theta_list:
            new_theta = torch.rand_like(theta)
            new_theta.requires_grad_(True)
            if self.USE_CUDA:
                new_theta = new_theta.cuda()
            new_theta_list.append(new_theta)
        return new_theta_list
    
    '''
    根据传入的theta_list，记录下每个theta的维度，以及拼接起来的话总维度是多少
    '''
    def theta_size(self, theta_list):
        self.theta_size_list = []
        self.total_theta_size = 0
        for theta in theta_list:
            size = list(theta.size())
            self.theta_size_list.append(size)
            num = reduce(lambda x, y : x * y, size)
            self.total_theta_size += num
    
    '''
    根据theta拼接起来后的总长度self.total_theta_size，创建state_h, state_c，合并成一个list，记为state_loader
    '''
    def state_cat(self):
        state_h = torch.zeros(self.num_stacks, self.total_theta_size, self.hidden_size)
        state_c = torch.zeros(self.num_stacks, self.total_theta_size, self.hidden_size)
        if self.USE_CUDA:
            state_h = state_h.cuda()
            state_c = state_c.cuda()
        state_loader = [state_h, state_c]
        return state_loader
    
    '''
    loss.backward()后，theta_list中会产生梯度，将梯度展开后拼接成grad_loader，维度是[self.total_theta_size, 1]
    '''
    def grad_cat(self, theta_list):
        grad_loader = torch.Tensor([])
        if self.USE_CUDA:
            grad_loader = grad_loader.cuda()
        grad_loader.requires_grad_(True)
        for theta in theta_list:
            grad = theta.grad.clone().detach().reshape((-1, 1))
            
            grad_loader = torch.cat((grad_loader, grad), dim=0)

        return grad_loader
    
    '''
    将LSTM返回的update_batch一个一个拼接起来，最终存为和grad_loader维度相同的update_loader
    同时用LSTM返回的cur_state_batch，更新原来的state_loader
    '''
    def update_cat(self, update_batch, cur_state_batch, update_loader, state_loader):
        start = self.prev_batch_start_index
        end = self.batch_start_index
        update_loader = torch.cat((update_loader, update_batch), dim=0)
        state_loader[0][:, start : end, :] = cur_state_batch[0][:, :, :]
        state_loader[1][:, start : end, :] = cur_state_batch[1][:, :, :]
        self.prev_batch_start_index = self.batch_start_index
        return update_loader, state_loader
    
    '''
    将update_loader按照theta_list里每个theta的维度，加到theta上，最终返回新theta_list
    '''
    def theta_update(self, update_loader, theta_list):
        
        start = 0
        for i in range(len(theta_list)):
            size = self.theta_size_list[i]
            num = reduce(lambda x, y : x * y, size)
            update = torch.reshape(update_loader[start : start + num, :], tuple(size))
            update.requires_grad_(True)
            
            theta_list[i] = torch.add(theta_list[i], update)
            theta_list[i].retain_grad()
            update.retain_grad()
            
            start += num
        return theta_list
    
    '''
    LSTM优化好后在新model上检验，循环调用step(model)来更新model里的参数，这里便是将LSTM计算出的update加到model参数上
    '''
    def model_update(self, update_loader, model):
        start = 0
        for i, p in enumerate(model.parameters()):
            size = self.theta_size_list[i]
            num = reduce(lambda x, y : x * y, size)
            update = torch.reshape(update_loader[start : start + num, :], tuple(size))
            p.detach_()
            p.add_(update)
            start += num
    
    '''
    从grad_loader和state_loader中循环地取出一个batch，作为LSTM的输入
    '''
    def get_batch(self, grad_loader, state_loader):
        total_size = list(grad_loader.size())[0]
        assert total_size == self.total_theta_size
        end_index = self.batch_start_index + self.batch_size
        if end_index < total_size:
            grad_batch = grad_loader[self.batch_start_index : end_index, :]
            state_batch = (state_loader[0][:, self.batch_start_index : end_index, :].contiguous(),
                           state_loader[1][:, self.batch_start_index : end_index, :].contiguous())
            
            self.batch_start_index = end_index
        else:
            grad_batch = grad_loader[self.batch_start_index : total_size, :]
            state_batch = (state_loader[0][:, self.batch_start_index : total_size, :].contiguous(),
                           state_loader[1][:, self.batch_start_index : total_size, :].contiguous())
            
            self.batch_start_index = total_size
            self.batch_end_flag = True

        return grad_batch, state_batch
    
    '''
    画出一个UNROLL_ITERATION中，MODEL_LOSS随epoch的下降趋势
    '''
    def plot_model_loss(self, k):
        step = np.arange(self.UNROLL_ITERATION)
        plt.plot(step, self.MODEL_LOSS_HIST)
        plt.xlabel("Unroll Iteration")
        plt.ylabel("Loss")
        plt.title(" Unroll Model Loss")
        plt.savefig("Unroll_Model_Loss_"+str(k+1)+".png", format="png")
        plt.cla()
        plt.clf()
    
    '''
    画出LSTM_TRAIN_ITERATION中，LSTM_TRAIN_LOSS随k的下降趋势
    '''
    def plot_lstm_loss(self):
        step = np.arange(self.LSTM_TRAIN_ITERATION)
        plt.plot(step, self.LSTM_TRAIN_LOSS_HIST)
        plt.xlabel("LSTM Train Iteration")
        plt.ylabel("Loss")
        plt.title("LSTM Train Loss")
        plt.savefig("LSTM_Train_Loss.png", format="png")
        plt.cla()
        plt.clf()

    '''
    learn()这个函数可能需要修改，将被优化的参数们写成一个包含许多tensor的list，存为model_theta_list，作为learn()的输入
    同时，model不再是nn.Module的子类，而是一个model_theta_list的函数，outputs = model(model_theta_list, inputs)
    '''
    def learn(self, model_theta_list, model, criterion, data_loader):
        self.lstm_reset()
        theta_list = copy.deepcopy(model_theta_list)
        if self.USE_CUDA:
            for i in range(len(theta_list)):
                theta_list[i] = theta_list[i].cuda()
        self.theta_size(theta_list)
        best_lstm_train_loss = 99999.0
        best_lstm_model = copy.deepcopy(self.lstm_model)

        for k in range(self.LSTM_TRAIN_ITERATION):
            self.LSTM_TRAIN_LOSS = 0
            self.MODEL_LOSS_HIST = []
            if k % self.THETA_RESET_INTERVAL == 0:
                theta_list = self.theta_reset(theta_list)
            update_loaders = []
            state_loader = self.state_cat()
            self.zero_grad(self.lstm_model.parameters())
            for epoch in range(self.UNROLL_ITERATION):
                print(epoch)
                
                self.zero_grad(theta_list)
                
                MODEL_LOSS = 0
                for inputs, targets in data_loader:
                    if self.USE_CUDA:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    model.load_params(theta_list)
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, targets)
                    MODEL_LOSS += loss
                self.LSTM_TRAIN_LOSS += MODEL_LOSS
                self.MODEL_LOSS_HIST.append(MODEL_LOSS.item())
        
                MODEL_LOSS.backward(retain_graph=True)
                
                grad_loader = self.grad_cat(theta_list)
                
                self.batch_start_index = 0
                self.prev_batch_start_index = 0
                self.batch_end_flag = False
                update_loaders.append(torch.Tensor([]))
                if self.USE_CUDA:
                    update_loaders[epoch] = update_loaders[epoch].cuda()
                update_loaders[epoch].requires_grad_(True)
                while True:
                    grad_batch, state_batch = self.get_batch(grad_loader, state_loader)
                    update_batch, cur_state_batch = self.lstm_model(grad_batch, state_batch)
                    update_loaders[epoch], state_loader = self.update_cat(update_batch, cur_state_batch, 
                                                                          update_loaders[epoch], state_loader)
                    if self.batch_end_flag == True:
                        break
                
                theta_list = self.theta_update(update_loaders[epoch], theta_list)
                #print("new theta list",theta_list[0].shape,theta_list[1].shape,theta_list[2].shape,theta_list[3].shape)
                state_loader = [state_loader[0].detach(), state_loader[1].detach()]

            # 代码彻底跑通后再画这个图
            # self.plot_model_loss(k)
            # 不知道是否需要这个大LOSS.backward()，还得测试
            #self.LSTM_TRAIN_LOSS.backward(retain_graph=True)
            self.LSTM_ADAM.step()
            # 观察LSTM里的参数是否随着k的增加一直迭代，也可以把p.grad打印出来
            for i, p in enumerate(self.lstm_model.lstm.parameters()):
                if i == 0:
                    print("lstm param")
                    print(p)
            self.LSTM_TRAIN_LOSS_HIST.append(self.LSTM_TRAIN_LOSS.item())
            print("lstm train step : %i / %i" %(k+1, self.LSTM_TRAIN_ITERATION))
            print("lstm train loss : %.6f"    %self.LSTM_TRAIN_LOSS.item())
            print("-" * 30 + "\n")
            if self.LSTM_TRAIN_LOSS.item() < best_lstm_train_loss:
                best_lstm_train_loss = self.LSTM_TRAIN_LOSS.item()
                best_lstm_model = copy.deepcopy(self.lstm_model)
        
        self.plot_lstm_loss()
        self.lstm_model = copy.deepcopy(best_lstm_model)
        torch.save(self.lstm_model.state_dict(), "./lstm_model.pkl")
        
        
    def continue_learn(self, model_theta_list, model, criterion, data_loader):
        #self.lstm_reset()
        self.lstm_model.load_state_dict(torch.load("./lstm_model.pkl"))
        theta_list = copy.deepcopy(model_theta_list)
        if self.USE_CUDA:
            for i in range(len(theta_list)):
                theta_list[i] = theta_list[i].cuda()
        self.theta_size(theta_list)
        best_lstm_train_loss = 99999.0
        best_lstm_model = copy.deepcopy(self.lstm_model)

        for k in range(self.LSTM_TRAIN_ITERATION):
            self.LSTM_TRAIN_LOSS = 0
            self.MODEL_LOSS_HIST = []
            if k % self.THETA_RESET_INTERVAL == 0:
                theta_list = self.theta_reset(theta_list)
            update_loaders = []
            state_loader = self.state_cat()
            self.zero_grad(self.lstm_model.parameters())
            for epoch in range(self.UNROLL_ITERATION):
                print(epoch)

                self.zero_grad(theta_list)

                MODEL_LOSS = 0
                for inputs, targets in data_loader:
                    if self.USE_CUDA:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    model.load_params(theta_list)
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, targets)
                    MODEL_LOSS += loss
                self.LSTM_TRAIN_LOSS += MODEL_LOSS
                self.MODEL_LOSS_HIST.append(MODEL_LOSS.item())

                MODEL_LOSS.backward(retain_graph=True)

                grad_loader = self.grad_cat(theta_list)

                self.batch_start_index = 0
                self.prev_batch_start_index = 0
                self.batch_end_flag = False
                update_loaders.append(torch.Tensor([]))
                if self.USE_CUDA:
                    update_loaders[epoch] = update_loaders[epoch].cuda()
                update_loaders[epoch].requires_grad_(True)
                while True:
                    grad_batch, state_batch = self.get_batch(grad_loader, state_loader)
                    update_batch, cur_state_batch = self.lstm_model(grad_batch, state_batch)
                    update_loaders[epoch], state_loader = self.update_cat(update_batch, cur_state_batch,
                                                                          update_loaders[epoch], state_loader)
                    if self.batch_end_flag == True:
                        break

                theta_list = self.theta_update(update_loaders[epoch], theta_list)
                # print("new theta list",theta_list[0].shape,theta_list[1].shape,theta_list[2].shape,theta_list[3].shape)
                state_loader = [state_loader[0].detach(), state_loader[1].detach()]

            # 代码彻底跑通后再画这个图
            # self.plot_model_loss(k)
            # 不知道是否需要这个大LOSS.backward()，还得测试
            # self.LSTM_TRAIN_LOSS.backward(retain_graph=True)
            self.LSTM_ADAM.step()
            # 观察LSTM里的参数是否随着k的增加一直迭代，也可以把p.grad打印出来
            for i, p in enumerate(self.lstm_model.lstm.parameters()):
                if i == 0:
                    print("lstm param")
                    print(p)
            self.LSTM_TRAIN_LOSS_HIST.append(self.LSTM_TRAIN_LOSS.item())
            print("lstm train step : %i / %i" % (k + 1, self.LSTM_TRAIN_ITERATION))
            print("lstm train loss : %.6f" % self.LSTM_TRAIN_LOSS.item())
            print("-" * 30 + "\n")
            if self.LSTM_TRAIN_LOSS.item() < best_lstm_train_loss:
                best_lstm_train_loss = self.LSTM_TRAIN_LOSS.item()
                best_lstm_model = copy.deepcopy(self.lstm_model)

        self.plot_lstm_loss()
        self.lstm_model = copy.deepcopy(best_lstm_model)
        torch.save(self.lstm_model.state_dict(), "./lstm_model.pkl")

   
    def init_step(self, model_params):
        self.lstm_model.load_state_dict(torch.load("./lstm_model.pkl"))
        self.theta_size(model_params)
        self.state_loader = self.state_cat()
    
    def step(self, model):
        grad_loader = self.grad_cat(model.parameters())
        self.batch_start_index = 0
        self.prev_batch_start_index = 0
        self.batch_end_flag = False
        update_loader = torch.Tensor([])
        if self.USE_CUDA:
            update_loader = update_loader.cuda()
        while True:
            grad_batch, state_batch = self.get_batch(grad_loader, self.state_loader)
            update_batch, cur_state_batch = self.lstm_model(grad_batch, state_batch)
            update_loader, self.state_loader = self.update_cat(update_batch, cur_state_batch, 
                                                            update_loader, self.state_loader)
            if self.batch_end_flag == True:
                break
        self.model_update(update_loader, model)
