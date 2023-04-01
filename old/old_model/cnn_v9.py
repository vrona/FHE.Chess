import torch
import torch.nn as nn
import torch.nn.functional as F

"""
w: input volume size
F: kernel/filter size
P: amount of zero padding
S: stride
output_w : (W-F+2P)/S+1
eg.: if input 7*7 / filter 3*3 / s 1 / pad 0 then output_w 3*3

8-3+2
"""

class Net(nn.Module):

    def __init__(self, hidden_size):

        super(Net, self).__init__()
        # define layers of CNN

        # input >> hidden layer
        self.conv1 = nn.Conv2d(hidden_size, hidden_size,kernel_size=3, stride=1, padding=1)
        self.batchn1 = nn.BatchNorm2d(hidden_size)

        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.batchn2 = nn.BatchNorm2d(hidden_size)

    def forward(self, x):
        # define forward behavior
        x_input = torch.clone(x)

        # activations and batch normalization
        x = self.conv1(x)
        x = self.batchn1(x)
        x = F.selu(x)

        x = self.conv2(x)
        x = self.batchn2(x)

        x = x + x_input
        x = F.selu(x)

        return x


class PlainChessNET(nn.Module):

    def __init__(self, hidden_layers=2, hidden_size=384):

        super(PlainChessNET, self).__init__()
        
        # define layers of CNN

        # input >> hidden layer
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, kernel_size=3, stride=1, padding=1)
        self.modulelist = nn.ModuleList([Net(hidden_size) for i in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 128, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(8192, 128)
        self.batchn1d_1 = nn.BatchNorm1d(128)
        self.output = nn.Linear(128,64)


    def forward(self, x):
        # define forward behavior

        # add sequence of convolutional
        #x = F.max_pool2d()
        x = self.input_layer(x)
        x = F.relu(x)

        for h in range(self.hidden_layers):
            x = self.modulelist[h](x)
        
        x = self.output_layer(x) # torch.Size([32, 2, 8, 8])
        x = self.flatten(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.output(x)

        return x