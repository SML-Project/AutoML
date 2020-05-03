import torch
import numpy as np
from SetParam.py import SetParam
from LoadData.py import LoadData
from BuildModel.py import BuildModel
from LSTMModel.py import LSTMModel
from LSTMLearner.py import LSTMLearner
from TrainModel.py import TrainModel
from ContrastLearn.py import ConstrastLearn
from PlotCurve.py import PlotCurve


def _main():
    ParamDict = SetParam()
    DataDict = LoadData()
    model, device = BuildModel()
    lstm_model = LSTMModel()
    lstm_learner = LSTMLearner()
    loss_hist_lstm = TrainModel()
    loss_hist_contrast = ContrastLearn()
    PlotCurve()


if __name__ == "__main__":
    _main()