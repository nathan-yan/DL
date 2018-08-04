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
    
    hidden_states, cell_states = generateState(layers)

    lstm1 = LSTM(OUTPUT_SIZE, HIDDEN_SIZE)
    lstm2 = LSTM(HIDDEN_SIZE, HIDDEN_SIZE)
    lstm3 = LSTM(HIDDEN_SIZE, HIDDEN_SIZE)
    lstm4 = LSTM(HIDDEN_SIZE, OUTPUT_SIZE)

    cumulative_loss = 0

    parameters = []
    for layer in [lstm1, lstm2, lstm3, lstm4]:
        for p in layer.parameters():
            parameters.append(p)

    optimizer = optim.RMSprop(parameters, lr = 0.001 )    

    gpu_count = torch.cuda.device_count()
    print("There are %d GPUs available!" % gpu_count)

    if (not gpu_count):
        print("Using CPU...")
    else:
        lstm1.cuda(device)
        lstm2.cuda(device)
        lstm3.cuda(device)
        lstm4.cuda(device)

        h1.cuda(device)
        c1.cuda(device)
        
        h2.cuda(device)
        c2.cuda(device)
        
        h3.cuda(device)
        c3.cuda(device)

        h4.cuda(device)
        c4.cuda(device)

    print(h1)

    for epoch in range (100000):
        h1, c1 = torch.zeros([1, HIDDEN_SIZE], device = device), torch.zeros([1, HIDDEN_SIZE], device = device)
        h2, c2 = torch.zeros([1, HIDDEN_SIZE], device = device), torch.zeros([1, HIDDEN_SIZE], device = device)
        h3, c3 = torch.zeros([1, HIDDEN_SIZE], device = device), torch.zeros([1, HIDDEN_SIZE], device = device)
        h4, c4 = torch.zeros([1, OUTPUT_SIZE], device = device), torch.zeros([1, OUTPUT_SIZE], device = device)

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

            h1, c1 = lstm1.forward(onehot, h1, c1)
            h2, c2 = lstm2.forward(h1, h2, c2)
            h3, c3 = lstm3.forward(h2, h3, c3)
            h4, c4 = lstm4.forward(h3, h4, c4)

            output = nn.Softmax(dim = 1)(h4) 

            loss = -torch.log(torch.sum(output * target))
            #print(output, target)
            cumulative_loss += loss

            if c % 50 == 0 and c != 0:
                print("LOSS -", torch.mean(cumulative_loss))
                average_loss = torch.mean(cumulative_loss)

                average_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                cumulative_loss = 0

                h1, c1 = torch.tensor(h1.data, device = device), torch.tensor(c1.data, device = device)
                h2, c2 = torch.tensor(h2.data, device = device), torch.tensor(c2.data, device = device)
                h3, c3 = torch.tensor(h3.data, device = device), torch.tensor(c3.data, device = device)
                h4, c4 = torch.tensor(h4.data, device = device), torch.tensor(c4.data, device = device)

                h1.cuda(device)
                c1.cuda(device)
                
                h2.cuda(device)
                c2.cuda(device)
                
                h3.cuda(device)
                c3.cuda(device)

                h4.cuda(device)
                c4.cuda(device)
            
            if c % 8000 == 0:
                h1, c1 = torch.zeros([1, HIDDEN_SIZE], device = device), torch.zeros([1, HIDDEN_SIZE], device = device)
                h2, c2 = torch.zeros([1, HIDDEN_SIZE], device = device), torch.zeros([1, HIDDEN_SIZE], device = device)
                h3, c3 = torch.zeros([1, HIDDEN_SIZE], device = device), torch.zeros([1, HIDDEN_SIZE], device = device)
                h4, c4 = torch.zeros([1, OUTPUT_SIZE], device = device), torch.zeros([1, OUTPUT_SIZE], device = device)

                h1.cuda(device)
                c1.cuda(device)
                
                h2.cuda(device)
                c2.cuda(device)
                
                h3.cuda(device)
                c3.cuda(device)

                h4.cuda(device)
                c4.cuda(device)
             
            if c % 8000 == 0:
                # Randomly sample from the network to see its progress
                sample([lstm1, lstm2, lstm3, lstm4], init = 0)

    # lstm1 = LSTM(10, 100)

    

if __name__ == "__main__":
    main()