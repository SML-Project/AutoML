import torch
import copy
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import reduce


class LSTMModel(nn.Module):
    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
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
        self.LSTM_ADAM = torch.optim.Adam(self.lstm_model.parameters(), lr=self.LSTM_ADAM_LR,
                                          betas=self.LSTM_ADAM_BETAS, eps=self.LSTM_ADAM_EPS,
                                          weight_decay=self.LSTM_ADAM_WD)
        self.LSTM_TRAIN_LOSS = 0
        self.LSTM_TRAIN_LOSS_HIST = []
        if self.USE_CUDA:
            self.lstm_model = self.lstm_model.cuda()

    def zero_grad(self, model):
        for param in model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def weight_reset(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    
    def lstm_reset(self):
        self.LSTM_TRAIN_LOSS = 0
        self.LSTM_TRAIN_LOSS_HIST = []
        self.lstm_model.lstm.apply(self.weight_reset)

    def theta_reset(self, model):
        model.apply(self.weight_reset)
        for param in model.parameters():
            param.detach_()
            param.requires_grad_(True)
    
    def theta_size(self, model_params):
        theta_size = []
        for param in model_params:
            size = list(param.size())
            theta_size.append(size)
        return theta_size
    
    def theta_cat(self, model_params):
        param_loader = torch.Tensor([])
        if self.USE_CUDA:
            param_loader = param_loader.cuda()
        for p in model_params:
            param = p.clone().detach_().reshape((-1, 1))
            param_loader = torch.cat((param_loader, param), dim=0)
        param_loader.requires_grad_(True)
        param_loader.retain_grad()
        return param_loader
    
    def state_cat(self, param_loader):
        total_size = list(param_loader.size())[0]
        if self.USE_CUDA:
            state_loader = [torch.zeros(self.num_stacks, total_size, self.hidden_size).cuda(),
                            torch.zeros(self.num_stacks, total_size, self.hidden_size).cuda()]
        else:
            state_loader = [torch.zeros(self.num_stacks, total_size, self.hidden_size),
                            torch.zeros(self.num_stacks, total_size, self.hidden_size)]
        return state_loader
    
    def grad_cat(self, model):
        grad_loader = torch.Tensor([])
        if self.USE_CUDA:
            grad_loader = grad_loader.cuda()
        for p in model.parameters():
            grad = p.grad.clone().detach().reshape((-1, 1))
            grad_loader = torch.cat((grad_loader, grad), dim=0)
        return grad_loader
    
    def theta_split(self, param_loader, model):
        index = 0
        for i, p in enumerate(model.parameters()):
            size = self.theta_size_list[i]
            num = reduce(lambda x, y : x * y, size)
            param = param_loader[index : index+num, :].reshape(tuple(size))
            p = param.clone()
            p.requires_grad_(True)
            p.retain_grad()
            index += num

    def theta_update(self, update_batch, cur_state_batch, param_loader, state_loader):
        start = self.prev_batch_start_index
        end = self.batch_start_index
        param_loader[start : end, :] = param_loader[start : end, :] + update_batch[:, :]
        state_loader[0][:, start : end, :] = cur_state_batch[0][:, :, :]
        state_loader[1][:, start : end, :] = cur_state_batch[1][:, :, :]
        self.prev_batch_start_index = self.batch_start_index
    
    def get_batch(self, grad_loader, state_loader):
        total_size = list(grad_loader.size())[0]
        end_index = self.batch_start_index + self.batch_size
        if end_index < total_size:
            grad_batch = grad_loader[self.batch_start_index : end_index, :]
            state_batch = (state_loader[0][:, self.batch_start_index : end_index, :],
                           state_loader[1][:, self.batch_start_index : end_index, :])
            self.batch_start_index = end_index
        else:
            grad_batch = grad_loader[self.batch_start_index : total_size, :]
            state_batch = (state_loader[0][:, self.batch_start_index : total_size, :],
                           state_loader[1][:, self.batch_start_index : total_size, :])
            self.batch_start_index = total_size
            self.batch_end_flag = True
        return grad_batch, state_batch
    
    def plot_model_loss(self, k):
        step = np.arange(self.UNROLL_ITERATION)
        plt.plot(step, self.MODEL_LOSS_HIST)
        plt.xlabel("Unroll Iteration")
        plt.ylabel("Loss")
        plt.title(" Unroll Model Loss")
        plt.savefig("Unroll_Model_Loss_"+str(k+1)+".png", format="png")
        plt.cla()
        plt.clf()
    
    def plot_lstm_loss(self):
        step = np.arange(self.LSTM_TRAIN_ITERATION)
        plt.plot(step, self.LSTM_TRAIN_LOSS_HIST)
        plt.xlabel("LSTM Train Iteration")
        plt.ylabel("Loss")
        plt.title("LSTM Train Loss")
        plt.savefig("LSTM_Train_Loss.png", format="png")
        plt.cla()
        plt.clf()
    
    def learn(self, model_orig, criterion, data_loader):
        self.LSTM_TRAIN_LOSS_HIST = []
        self.lstm_reset()
        model = copy.deepcopy(model_orig)
        if self.USE_CUDA:
            model = model.cuda()
        self.theta_size_list = self.theta_size(model.parameters())
        best_lstm_train_loss = 99999.0
        best_lstm_model = copy.deepcopy(self.lstm_model)

        for k in range(self.LSTM_TRAIN_ITERATION):
            self.LSTM_TRAIN_LOSS = 0
            self.MODEL_LOSS_HIST = []
            if k % self.THETA_RESET_INTERVAL == 0:
                self.theta_reset(model)
            param_loader = self.theta_cat(model.parameters())
            state_loader = self.state_cat(param_loader)
            for epoch in range(self.UNROLL_ITERATION):
                self.zero_grad(model)
                self.theta_split(param_loader, model)
                MODEL_LOSS = 0
                for inputs, targets in data_loader["train"]:
                    if self.USE_CUDA:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    MODEL_LOSS += loss
                self.MODEL_LOSS_HIST.append(MODEL_LOSS.item())
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

            self.plot_model_loss(k)
            self.LSTM_ADAM.zero_grad()
            self.LSTM_TRAIN_LOSS.backward()
            self.LSTM_ADAM.step()
            self.LSTM_TRAIN_LOSS_HIST.append(self.LSTM_TRAIN_LOSS.item())
            print("lstm train step : %i / %i" %(k+1, self.LSTM_TRAIN_ITERATION))
            print("lstm train loss : %.6f"    %self.LSTM_TRAIN_LOSS.item())
            print("-" * 20 + "\n")
            if self.LSTM_TRAIN_LOSS.item() < best_lstm_train_loss:
                best_lstm_train_loss = self.LSTM_TRAIN_LOSS.item()
                best_lstm_model = copy.deepcopy(self.lstm_model)
        
        self.plot_lstm_loss()
        self.lstm_model = copy.deepcopy(best_lstm_model)

    def init_step(self, model_params):
        self.theta_size_list = self.theta_size(model_params)
        self.param_loader = self.theta_cat(model_params)
        self.state_loader = self.state_cat(self.param_loader)
    
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
        for p in model.parameters():
            p.detach_()
            p.requires_grad_(True)


'''
怎么用LSTMLearner 

learner = LSTMLearner(args)
learner.learn(train_model, criterion, data_loader)

def TrainModel():
    learner.init_step(trans_model.parameters())
    for epoch in step:
        learner.zero_grad(trans_model)
        loss.backward()
        learner.step(trans_model)
'''


