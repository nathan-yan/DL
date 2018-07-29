import pickle

import matplotlib.pyplot as plt 
import numpy as np

from sklearn.preprocessing import OneHotEncoder 

import torch
import torch.nn as nn

class CIFAR10(nn.Module):
    def __init__(self, input_shape, classes):
        super(CIFAR10, self).__init__()

        self.input_shape = input_shape
        self.classes = classes

        self.conv_1 = nn.Conv2d(
            in_channels = self.input_shape[1],
            out_channels = 32,
            kernel_size = 5,
            padding = 2
        )
        self.conv_1_relu = nn.ReLU()

        self.conv_2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64, 
            kernel_size = 3,
            padding = 1
        )
        self.conv_2_relu = nn.ReLU()

        self.pool_1 = nn.MaxPool2d(
            kernel_size = 2
        )

        self.conv_3 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 3,
            padding = 1 
        )
        self.conv_3_relu = nn.ReLU()

        self.conv_4 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.conv_4_relu = nn.ReLU()

        self.pool_2 = nn.MaxPool2d(
            kernel_size = 2
        )

        self.linear_1 = nn.Linear(128 * 8 * 8, 1000)
        self.linear_1_relu = nn.ReLU()

        self.linear_2 = nn.Linear(1000, self.classes)

        self.output = nn.Softmax(dim = 1)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_1_relu = self.conv_1_relu(conv_1)

        conv_2 = self.conv_2(conv_1_relu)
        conv_2_relu = self.conv_2_relu(conv_2)

        pool_1 = self.pool_1(conv_2_relu)

        conv_3 = self.conv_3(pool_1)
        conv_3_relu = self.conv_3_relu(conv_3)

        conv_4 = self.conv_4(conv_3_relu)
        conv_4_relu = self.conv_4_relu(conv_4)

        pool_2 = self.pool_2(conv_4_relu)

        flatten = pool_2.view(self.input_shape[0], -1)

        linear_1 = self.linear_1(flatten)
        linear_1_relu = self.linear_1_relu(linear_1)

        linear_2 = self.linear_2(linear_1_relu)

        output = self.output(linear_2)

        return output

def loadData(path = '../datasets/cifar10/'):
    prefix = path + 'data_batch_' 

    data = [] 
    targets = []

    for i in range (1, 6):
        filename = prefix + str(i)

        with open(filename, 'rb') as batch:
            d = pickle.load(batch, encoding = 'bytes')
            data.append(d[b'data'])
            targets.append(d[b'labels'])
    
    data = np.concatenate(data, axis = 0)
    targets = np.concatenate(targets, axis = 0)

    return data, targets

def onehot(a): 
    a = a.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse = False)
    onehot_encoded = onehot_encoder.fit_transform(a)

    return onehot_encoded

def main():
    #model = CIFAR10()

    data, targets = loadData()
    data, targets = torch.tensor(data, dtype = torch.float32)/255., torch.tensor(onehot(targets), dtype = torch.float32)
    print("Loaded data!")

    data = data.view(-1, 3, 32, 32)

    epochs = 20
    batchsize = 16
    lr = 0.01

    avg_loss = 0
    avg_acc = 0
    check_every = 32 * 20

    time = []
    loss_record = []
    acc_record = []
    iterations = 0

    model = CIFAR10(
        input_shape = [batchsize, 3, 32, 32],
        classes = 10
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print("There are %d GPUs available!" % gpu_count)

    if (not gpu_count):
        print("Using CPU...")
    else:
        model.cuda(device)
    
        data = data.cuda()
        targets = targets.cuda()

    for epoch in range (epochs):
        print("\nSTARTING EPOCH %d\n" % (epoch + 1))

        if (epoch == 5):
            lr /= 10.
        elif (epoch == 15):
            lr /= 10.

        for batch in range (0, len(data), batchsize):
            if (batch + 16 >= len(data)):
                continue

            # Reset gradients
            model.zero_grad()

            target = targets[batch : batch + 16]
            output = model.forward(data[batch : batch + 16])

            log_loss = torch.mean(
                -torch.sum(target * torch.log(output), dim = 1)
            )

            log_loss.backward()     # Get gradients

            for p in model.parameters():
                p.data.add_(
                    p.grad.data * -lr
                )
        
            output = output.cpu().detach().numpy()
            predictions = np.argmax(output, axis = 1)

            batch_acc = np.mean(predictions == np.argmax(target.cpu().numpy(), axis = 1), axis = 0)

            avg_loss += log_loss 
            avg_acc += batch_acc

            if (batch % check_every == 0 and batch != 0):
                print("Loss - %f | Acc - %f" % (avg_loss/(check_every/batchsize), avg_acc/(check_every/batchsize)))

                iterations += 1
                time.append(iterations)
                loss_record.append(avg_loss/(check_every/batchsize))
                acc_record.append(avg_acc/(check_every/batchsize))

                avg_loss = 0
                avg_acc = 0

            if (batch % check_every * 5 == 0):
                plt.cla()
                plt.plot(time, loss_record, 'b-', time, acc_record, 'g-')
                plt.pause(0.00001)

    #print(model.forward(data[:32])[0], targets)

    """
        print(data[0].permute(2, 0, 1).size())
        plt.imshow(data[1].permute(1, 2, 0))
        plt.show()
    """

main()