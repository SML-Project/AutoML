import torch
import Models_tensor
from NewTrain import args,get_net_params,train_mnist_cifar_learner
from LSTMLearnerTensor import LSTMLearner
from LoadData import get_data_loader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if args.USE_CUDA else "cpu")
EPOCH_NUM=5
def test_base_optimizer(net, dataset_name, criterion, optimizer, args=args):
    t_loss = []
    #net=net.to(device)
    #for epoch in range(args.num_epochs):
    for epoch in range(EPOCH_NUM):
        MODEL_LOSS = 0
        train_loader = get_data_loader(dataset_name=dataset_name)
        print(epoch)
        optimizer.zero_grad()

        MODEL_LOSS = 0
        for batch_id, (data, target) in enumerate(train_loader):
            #print(batch_id, data.size(), target.size(), data.size()[1] * data.size()[2],len(train_loader.dataset), len(train_loader))
            data, target = data.to(device), target.to(device)
            output = net.forward(data)
            loss = criterion(output, target)
            #loss.backward()

            MODEL_LOSS += loss

            '''
            if batch_id % args.log_interval == 0:
                print(str(batch_id))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_id * len(data), len(train_loader.dataset),
                           100. * batch_id / len(train_loader), loss.item()))
            '''
        train_loss=MODEL_LOSS.item() / len(train_loader)
        t_loss.append(train_loss)
        MODEL_LOSS.backward()
        optimizer.step()
        if(epoch%20==0):
            print("loss:", train_loss)


    return t_loss

def test_learner(net,params,dataset_name, criterion, args=args):

    learner_a = LSTMLearner(args)
    data_loader = get_data_loader(dataset_name=dataset_name)

    print(len(data_loader))
    learner_a.lstm_model.load_state_dict(torch.load("./lstm_model.pkl"))
    if args.USE_CUDA:
        for i in range(len(params)):
            params[i] = params[i].cuda()
    learner_a.theta_size(params)
    # best_lstm_train_loss = 99999.0
    # best_lstm_model = copy.deepcopy(self.lstm_model)

    learner_a.LSTM_TRAIN_LOSS = 0
    learner_a.MODEL_LOSS_HIST = []
    # if k % self.THETA_RESET_INTERVAL == 0:
    #    theta_list = self.theta_reset(theta_list)
    update_loaders = []
    state_loader = learner_a.state_cat()
    for epoch in range(EPOCH_NUM):
        print(epoch)

        learner_a.zero_grad(params)

        MODEL_LOSS = 0
        for inputs, targets in data_loader:
            if args.USE_CUDA:
                inputs = inputs.cuda()
                targets = targets.cuda()
            net.load_params(params)
            outputs = net.forward(inputs)
            loss = criterion(outputs, targets)

            MODEL_LOSS += loss
        # learner_a.LSTM_TRAIN_LOSS += MODEL_LOSS
        learner_a.MODEL_LOSS_HIST.append(MODEL_LOSS.item() / len(data_loader))

        MODEL_LOSS.backward(retain_graph=True)

        grad_loader = learner_a.grad_cat(params)

        learner_a.batch_start_index = 0
        learner_a.prev_batch_start_index = 0
        learner_a.batch_end_flag = False
        update_loaders.append(torch.Tensor([]))
        if args.USE_CUDA:
            update_loaders[epoch] = update_loaders[epoch].cuda()
        update_loaders[epoch].requires_grad_(True)
        while True:
            grad_batch, state_batch = learner_a.get_batch(grad_loader, state_loader)
            update_batch, cur_state_batch = learner_a.lstm_model(grad_batch, state_batch)
            update_loaders[epoch], state_loader = learner_a.update_cat(update_batch, cur_state_batch,
                                                                       update_loaders[epoch], state_loader)
            if learner_a.batch_end_flag == True:
                break

        params = learner_a.theta_update(update_loaders[epoch], params)
        # print("new theta list",theta_list[0].shape,theta_list[1].shape,theta_list[2].shape,theta_list[3].shape)
        state_loader = [state_loader[0].detach(), state_loader[1].detach()]

        # print("params",params)
        if(epoch%10==0):
            print("loss:", MODEL_LOSS.item() / len(data_loader))
    t_loss=learner_a.MODEL_LOSS_HIST
    return t_loss

def test_script(net_type, dataset_name, criterion, optimizer, args=args):
    '''
    if net_type == "Qua":
        input_units = get_data_size()
        net = Models_tensor.Qua(input_units).to(device)
    '''
    filename=net_type+dataset_name+optimizer+".txt"
    net = Models_tensor.TwoLayer_Tensor(input_units=784, hidden_units=20)
    params = get_net_params(net)
    if net_type == "TwoLayerNet":
        net = Models_tensor.TwoLayer_Tensor(input_units=784, hidden_units=20)
        params = get_net_params(net)
    elif net_type == "ThreeLayerNet":
        net = Models_tensor.ThreeLayer_Tensor(input_units=784, hidden_units=20,activation1="sigmoid")
        params = get_net_params(net)
    '''
    elif net_type == "ConvMNISTNet":
        net = Models.ConvMNISTNet().to(device)
    elif net_type == "ConvCIFARNet":
        net = Models.ConvCIFARNet().to(device)
    '''
    opt_SGD = optim.SGD(params, lr=args.LR)
    opt_Momentum = optim.SGD(params, lr=args.LR, momentum=0.8)
    opt_RMSprop = optim.RMSprop(params, lr=args.LR, alpha=0.9)
    opt_Adam = optim.Adam(params, lr=args.LR, betas=(0.9, 0.99))
    opt_dict_set = {"opt_SGD": opt_SGD, "opt_Momentum": opt_Momentum, "opt_RMSprop": opt_RMSprop, "opt_Adam": opt_Adam}

    if optimizer == "LSTMLearner":
        t_loss=test_learner(net, params, dataset_name, criterion, args=args)
        torch.save(t_loss,filename)
        return t_loss

    for key, value in opt_dict_set.items():
        if optimizer == key:
            print(key)
            optimizer = opt_dict_set[key]
            t_loss=test_base_optimizer(net, dataset_name, criterion, optimizer, args=args)
            torch.save(t_loss,filename)
            return t_loss

def plot_compare_loss(learner_loss,sgd_loss,momentum_loss,rms_loss,adam_loss):
    step = np.arange(len(learner_loss))
    p1, = plt.plot(step, learner_loss, label='LSTMLearner')
    p2, = plt.plot(step, sgd_loss, label='SGD')
    p3, = plt.plot(step, momentum_loss, label='Momentum')
    p4, = plt.plot(step, rms_loss, label='RMS')
    p5, = plt.plot(step, adam_loss, label='Adam')
    plt.legend(handles=[p1, p2, p3, p4,p5])
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Compare Optimizer Losses")
    plt.savefig("Compare Optimizer Losses.png", format="png")
    plt.cla()
    plt.clf()
