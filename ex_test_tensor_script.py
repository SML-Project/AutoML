import torch
from NewTrain import args
import torch.nn as nn
from TestOptimizerTensor import test_script,plot_compare_loss

device = torch.device("cuda:0" if args.USE_CUDA else "cpu")

optimizers=["LSTMLearner","opt_SGD", "opt_Momentum", "opt_RMSprop", "opt_Adam"]

net_type="TwoLayer_Tensor"
dataset_name="MNIST"
criterion_name="CrossEntropy"
criterion = nn.CrossEntropyLoss()
Losses={}
for optimizer_name in optimizers:
    loss_history=test_script(net_type, dataset_name, criterion, optimizer=optimizer_name, args=args)
    Losses[optimizer_name]=loss_history
print(Losses)
plot_compare_loss(Losses["LSTMLearner"],Losses["opt_SGD"],Losses["opt_Momentum"],Losses["opt_RMSprop"],Losses["opt_Adam"])

