import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed", category = RuntimeWarning)

import pickle

import matplotlib.pyplot as plt 
import numpy as np

from sklearn.preprocessing import OneHotEncoder 

import torch
import torch.nn as nn
import torch.optim as optim

allowed_characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ[]{}-=_+0123456789!@#$%^&*()~`:?<>"\'/|;., \n\r\t'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def sample(layers, init, characters = 1000):
    hidden_states, cell_states = generateState(layers)

    for i in range (len(hidden_states)):
        hidden_states[i].cuda(device)
    
    for i in range (len(cell_states)):
        cell_states[i].cuda(device)

    prev_character = torch.zeros(1, layers[-1].hidden_size, device = device)
    prev_character[0][init] = 1

    acc = ''

    for c in range (characters):
        hidden_states[0], cell_states[0] = layers[0].forward(prev_character, hidden_states[0], cell_states[0])
    
        for i, l in enumerate(layers[1:]):
            i_ = i + 1
            hidden_states[i_], cell_states[i_] = l.forward(hidden_states[i], hidden_states[i_], cell_states[i_])

        output = nn.Softmax(dim = 1)(hidden_states[-1]/0.5)

        idx = np.random.choice(layers[-1].hidden_size, p = output[0].cpu().detach().numpy())

        prev_character = torch.zeros(1, layers[-1].hidden_size, device = device)
        prev_character[0][idx] = 1

        acc += allowed_characters[idx]
    
    print(acc)

def generateState(layers):
    hidden_states = [
        torch.zeros([1, layers[i].hidden_size], device = device) for i in range (len(layers))
    ]

    cell_states = [
        torch.zeros([1, layers[i].hidden_size], device = device) for i in range (len(layers))
    ]

    return hidden_states, cell_states

def main():
    char_to_idx = {}

    i = 0
    for c in allowed_characters:
        char_to_idx[c] = i
        i += 1

    HIDDEN_SIZE = 500
    OUTPUT_SIZE = len(allowed_characters)

    f = open('../datasets/linux_concat.txt', 'r')
    data = f.read()
    f.close()

    length = len(data)

    # The chaRNN processes the data 128 characters at a time, each iteration retaining the last hidden and cell of the previous iteration
    # Every 8192 characters the network completely resets
    
    layers = [
        LSTM(OUTPUT_SIZE, HIDDEN_SIZE),
        LSTM(HIDDEN_SIZE, HIDDEN_SIZE),
        LSTM(HIDDEN_SIZE, HIDDEN_SIZE),
        LSTM(HIDDEN_SIZE, OUTPUT_SIZE)
    ]

    cumulative_loss = 0

    parameters = []
    for layer in layers:
        for p in layer.parameters():
            parameters.append(p)

    optimizer = optim.RMSprop(parameters, lr = 0.001 )    

    gpu_count = torch.cuda.device_count()
    print("There are %d GPUs available!" % gpu_count)

    if (not gpu_count):
        print("Using CPU...")
    else:
        for l in layers:
            l.cuda(device)
            
    for epoch in range (100000):
        hidden_states, cell_states = generateState(layers)

        for c in range (length - 1):
            onehot = torch.zeros([1, OUTPUT_SIZE], dtype = torch.float32, device = device)
            char = char_to_idx.get(data[c])

            if not char:
                char = 0

            onehot[0][char] = 1

            target = torch.zeros([1, OUTPUT_SIZE], dtype = torch.float32, device = device)
            char = char_to_idx.get(data[c + 1])

            if not char:
                char = 0

            target[0][char] = 1

            onehot.cuda(device)
            target.cuda(device)

            for i, layer in enumerate(layers):
                if i == 0:
                    hidden_states[i], cell_states[i] = layer.forward(onehot, hidden_states[i], cell_states[i]) 
                else:
                    hidden_states[i], cell_states[i] = layer.forward(hidden_states[i - 1], hidden_states[i], cell_states[i]) 
            
            output = nn.Softmax(dim = 1)(hidden_states[-1]) 

            loss = -torch.log(torch.sum(output * target))
            #print(output, target)
            cumulative_loss += loss

            if c % 50 == 0 and c != 0:
                if c % 1000 == 0:
                    print("LOSS -", cumulative_loss / 50.)

                average_loss = cumulative_loss / 50.

                average_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                cumulative_loss = 0

                for l in range (len(layers)):
                    hidden_states[l], cell_states[l] = torch.tensor(hidden_states[l].data, device = device), torch.tensor(cell_states[l].data, device = device)

            if c % 8000 == 0:
                hidden_states, cell_states = generateState(layers)
             
            if c % 8000 == 0:
                # Randomly sample from the network to see its progress
                sample(layers, init = 0)

    # lstm1 = LSTM(10, 100)

    

if __name__ == "__main__":
    main()