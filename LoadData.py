import torch
import numpy as np

'''
a general method to load various type of data from external files
'''

def LoadData(N, batch_size, num_workers):
    X = torch.unsqueeze(torch.rand(N) * 4 - 2, dim=1)
    Y = X * (8 * X).sin() * (4 / X).sin()
    X_train, X_valid = X.split([int(0.7*N), int(0.3*N)], dim=0)
    Y_train, Y_valid = Y.split([int(0.7*N), int(0.3*N)], dim=0)
    data_train = TensorDataset(X_train, Y_train)
    data_valid = TensorDataset(X_valid, Y_valid)
    data = {"train": data_train, "valid": data_valid}
    for phase in ["train", "valid"]:
        print(phase + "_data: " + str(len(data[phase])))
    print("\n")
    loaders = {phase: DataLoader(data[phase], batch_size=batch_size, shuffle=True,  num_workers=num_workers)
                for phase in ["train", "valid"]}
    return loaders