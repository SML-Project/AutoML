import torch
import copy
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import reduce


class LSTMModel(nn.Module):
    def __init__(self, param_dict):
        super(LSTMModel, self).__init__()
        self.hidden_size = param_dict["hidden_size"] 
        self.batch_size = param_dict["batch_size"] 
        self.num_stacks = param_dict["num_stacks"]
        self.preprocess = param_dict["preprocess"]
        self.input_size = 2 if self.preprocess == True else 1
        self.output_size = 1
        self.p = param_dict["p"]
        self.output_scale = param_dict["output_scale"] 
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_stacks)
        self.Linear = nn.Linear(self.hidden_size, self.output_size)
        
    def gradient_preprocess(self, gradient):
        log = torch.log(torch.abs(gradient))
        clamp_log = torch.clamp(log/self.p, min=-1.0, max=1.0)
        clamp_sign = torch.clamp(torch.exp(torch.Tensor(self.p))*gradient, min=-1.0, max=1.0)
        gradient = torch.cat((clamp_log,clamp_sign), dim=-1)
        return gradient
    
    def gradient_update(self, gradient, prev_state):
        if prev_state is None:
            prev_state = (torch.zeros(self.num_stacks, self.batch_size, self.hidden_size),
                          torch.zeros(self.num_stacks, self.batch_size, self.hidden_size))
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


class LSTMLearner(object):
    def __init__(self, param_dict):
        self.lstm_model = LSTMModel(param_dict)
        self.hidden_size = param_dict["hidden_size"]
        self.batch_size = param_dict["batch_size"]
        self.num_stacks = param_dict["num_stacks"]
        self.USE_CUDA = param_dict["USE_CUDA"]
        self.LSTM_TRAIN_ITERATION = param_dict["LSTM_TRAIN_ITERATION"]
        self.UNROLL_ITERATION = param_dict["UNROLL_ITERATION"]
        self.THETA_RESET_INTERVAL = param_dict["THETA_RESET_INTERVAL"]
        self.LSTM_ADAM_LR = param_dict["LSTM_ADAM_LR"]
        self.LSTM_ADAM_BETAS = param_dict["LSTM_ADAM_BETAS"]
        self.LSTM_ADAM_EPS = param_dict["LSTM_ADAM_EPS"]
        self.LSTM_ADAM_WD = param_dict["LSTM_ADAM_WD"]
        self.LSTM_ADAM = torch.optim.Adam(self.lstm_model.paramters(), lr=self.LSTM_ADAM_LR,
                                          betas=self.LSTM_ADAM_BETAS, eps=self.LSTM_ADAM_EPS,
                                          weight_decay=self.LSTM_ADAM_WD)
        self.LSTM_TRAIN_LOSS = 0
        self.LSTM_TRAIN_LOSS_HIST = []
        if self.USE_CUDA:
            self.lstm_model = self.lstm_model.cuda()

    def zero_grad(self, model):
        for param in model.paramters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def weight_reset(m):
        if isinstance(m, nn.Module):
            m.reset_parameters()
    
    def lstm_reset(self):
        self.LSTM_TRAIN_LOSS = 0
        self.LSTM_TRAIN_LOSS_HIST = []
        self.lstm_model.apply(self.weight_reset)

    def theta_reset(self, model):
        model.apply(self.weight_reset)
        for param in model.parameters():
            param.detach_().requires_grad_(True)
    
    def theta_cat(self, model):
        param_loader = torch.Tensor([])
        for p in model.paramters():
            param = p.clone().reshape([-1, 1])
            param_loader = torch.cat((param_loader, param), dim=0)
        return param_loader
    
    def state_cat(self, param_loader):
        total_size = list(param_loader.size())[0]
        state_loader = [torch.zeros(self.num_stacks, total_size, self.hidden_size),
                        torch.zeros(self.num_stacks, total_size, self.hidden_size)]
        return state_loader
    
    def grad_cat(self, model):
        grad_loader = torch.Tensor([])
        for p in model.paramters():
            grad = p.grad.clone().detach_().reshape([-1, 1])
            grad_loader = torch.cat((grad_loader, grad), dim=0)
        return grad_loader
    
    def theta_split(self, param_loader, model):
        index = 0
        for i, p in enumerate(model.parameters()):
            size = self.theta_size_list[i]
            num = reduce(lambda x, y : x * y, size)
            p.detach_()
            param = param_loader[index : index+num, :].reshape(size)
            p.copy_(param)
            p.requires_grad_(True)
            index += num

    def theta_update(self, update_batch, cur_state_batch, param_loader, state_loader):
        start = self.prev_batch_start_index
        end = self.batch_start_index
        param_loader[start : end, :] = param_loader[start : end, :] + update_batch[:, :]
        state_loader[start : end, :] = state_batch[:, :]
        self.prev_batch_start_index = self.batch_start_index
    
    def get_batch(self, grad_loader, state_loader):
        total_size = list(grad_loader.size())[0]
        end_index = self.batch_start_index + self.batch_size
        if end_index < total_size:
            grad_batch = grad_loader[self.batch_start_index : end_index, :]
            state_batch = state_loader[self.batch_start_index : end_index, :]
            self.batch_start_index = end_index
        else if end_index == total_size:
            grad_batch = grad_loader[self.batch_start_index : end_index, :]
            state_batch = state_loader[self.batch_start_index : end_index, :]
            self.batch_start_index = end_index
            self.batch_end_flag = True
        else:
            diff = end_index - total_size
            grad_batch = grad_loader[self.batch_start_index : total_size, :]
            state_batch = state_loader[self.batch_start_index : total_size, :]
            for i in range(diff):
                grad_batch_diff = grad_loader[self.batch_start_index, :]
                state_batch_diff = state_loader[self.batch_start_index, :]
                grad_batch = torch.cat((grad_batch, grad_batch_diff), dim=0)
                state_batch = torch.cat((state_batch, state_batch_diff), dim=0)
            self.batch_start_index = total_size
            self.batch_end_flag = True
        return grad_batch, state_batch
    
    def plot_loss(self):
        step = np.arange(self.LSTM_TRAIN_ITERATION)
        plt.plot(step, self.LSTM_TRAIN_LOSS_HIST)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("LSTM Train Loss")
        plt.savefig("LSTM_Train_Loss.png", format="png")
        plt.cla()
        plt.clf()
    
    def learn(self, model_orig, criterion, data_loader):
        self.lstm_reset()
        if self.USE_CUDA:
            model = copy.deepcopy(model_orig.state_dict()).cuda()
        else:
            model = copy.deepcopy(model_orig.state_dict())
        self.theta_size_list = []
        for param in model.parameters():
            theta_size = list(param.size())
            self.theta_size_list.append(theta_size)
        best_lstm_train_loss = 999999
        best_lstm_model = copy.deepcopy(self.lstm_model.state_dict())

        for k in self.LSTM_TRAIN_ITERATION:
            self.LSTM_TRAIN_LOSS = 0
            if k & self.THETA_RESET_INTERVAL == 0:
                self.theta_reset(model)
            self.zero_grad(model)
            param_loader = self.theta_cat(model)
            state_loader = self.state_cat(param_loader)
            for epoch in self.UNROLL_ITERATION:
                self.theta_split(param_loader, model)
                MODEL_LOSS = 0
                for inputs, targets in data_loader["train"]:
                    if self.USE_CUDA:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    MODEL_LOSS += loss
                self.LSTM_TRAIN_LOSS += MODEL_LOSS
                MODEL_LOSS.backward(retain_graph=True)
                grad_loader = self.grad_cat(model)
                self.batch_start_index = 0
                self.prev_batch_start_index = 0
                self.batch_end_flag = False
                while True:
                    grad_batch, state_batch = self.get_batch(grad_loader, state_loader)
                    update_batch, cur_state_batch = self.lstm_model(grad_batch, state_batch)
                    self.theta_update(update_batch, cur_state_batch, param_loader, state_loader)
                    if self.batch_end_flag == True:
                        break
                param_loader.retain_grad()
                state_loader.retain_grad()

            self.LSTM_ADAM.zero_grad()
            self.LSTM_TRAIN_LOSS.backward()
            self.LSTM_ADAM.step()
            self.LSTM_TRAIN_LOSS_HIST.append(self.LSTM_TRAIN_LOSS.detach_().item())
            print("lstm train step : %i / %i" %(k+1, self.LSTM_TRAIN_ITERATION))
            print("lstm train loss : %f" %self.LSTM_TRAIN_LOSS)
            print("-" * 20 + "\n")
            if self.LSTM_TRAIN_LOSS < best_lstm_train_loss:
                best_lstm_train_loss = self.LSTM_TRAIN_LOSS
                best_lstm_model = copy.deepcopy(self.lstm_model.state_dict())
        
        self.plot_loss()
        self.lstm_model = copy.deepcopy(best_lstm_model)

    def init_step(self, model):
        self.param_loader = self.theta_cat(model)
        self.state_loader = self.state_cat(param_loader)
    
    def step(self, model):
        self.grad_loader = self.grad_cat(model)
        self.batch_start_index = 0
        self.prev_batch_start_index = 0
        self.batch_end_flag = False
        while True:
            grad_batch, state_batch = self.get_batch(self.grad_loader, self.state_loader)
            update_batch, cur_state_batch = self.lstm_model(grad_batch, state_batch)
            self.theta_update(update_batch, cur_state_batch, self.param_loader, self.state_loader)
            if self.batch_end_flag == True:
                break
        self.theta_split(self.param_loader, model)