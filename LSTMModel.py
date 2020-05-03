import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_size, num_stacks=2, 
                preprocess=True, p=10, output_scale=0.1):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_stacks = num_stacks
        self.preprocess = preprocess
        self.input_scale = 1
        if self.preprocess == True:
            self.input_scale = 2
        self.lstm = nn.LSTM(input_size*self.input_scale, hidden_size, num_stacks)
        self.Linear = nn.Linear(hidden_size, output_size)
        self.p = p
        self.output_scale = output_scale 
        
    def GradientPreprocess(self, gradients):
        log = torch.log(torch.abs(gradients))
        clamp_log = torch.clamp(log/p, min=-1.0, max=1.0)
        clamp_sign = torch.clamp(torch.exp(torch.Tensor(p))*gradients, min=-1.0, max=1.0)
        gradients = torch.cat((clamp_log,clamp_sign), dim=-1)
        return gradients
    
    def GradientUpdate(self, gradients, prev_state):
        if prev_state is None:
            prev_state = (torch.zeros(self.num_stacks, self.batch_size, self.hidden_size),
                          torch.zeros(self.num_stacks, self.batch_size, self.hidden_size))
        update, cur_state = self.lstm(gradients, prev_state)
        update = self.Linear(update) * self.output_scale 
        return update, cur_state
    
    def forward(self, gradients, prev_state):
        gradients = gradients.unsqueeze(0)
        if self.preprocess == True:
            gradients = self.GradientPreprocess(gradients)
        update, cur_state = self.GradientUpdate(gradients, prev_state)
        update = update.squeeze().squeeze()
        return update, cur_state

