import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed", category = RuntimeWarning)

import pickle

import matplotlib.pyplot as plt 
import numpy as np

from sklearn.preprocessing import OneHotEncoder 

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size 

        self.concat_size = self.input_size + self.hidden_size

        self.forget = nn.Linear(self.concat_size, self.hidden_size)
        self.input  = nn.Linear(self.concat_size, self.hidden_size)
        self.output = nn.Linear(self.concat_size, self.hidden_size)
        self.cell   = nn.Linear(self.concat_size, self.hidden_size)

    def forward(self, input, prev_hidden, prev_cell):
        # input, prev_hidden -> bs x size

        concat = torch.cat([input, prev_hidden], dim = 1)

        forget = nn.Sigmoid()(self.forget(concat))
        input  = nn.Sigmoid()(self.input(concat))
        output = nn.Sigmoid()(self.output(concat))
        cell   = nn.Tanh()(self.cell(concat))

        hidden = (forget * prev_cell + input * cell) * output 

        return hidden, cell 
    
def main():
    lstm = LSTM(10, 100)

    i = torch.ones(2, 10)
    h = torch.zeros(2, 100)
    c = torch.zeros(2, 100)

    print(lstm.forward(i, h, c)[0].shape)
    h, c = lstm.forward(i, h, c)
    print(h)
    h, c = lstm.forward(i, h, c)
    print(h)

if __name__ == "__main__":
    main()