import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

"""
w: input volume size
F: kernel/filter size
P: amount of zero padding
S: stride
output_w : (W−F+2P)/S+1
eg.: if input 7*7 / filter 3*3 / s 1 / pad 0 then output_w 3*3
"""

class Net(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers, drop_p = 0.5) -> None:
        super(Net, self).__init__()
        # define layers of CNN

        # input >> hidden layer
        self.conv1 = nn.Conv2d(8,8,12, padding=0)
        self.conv2 = nn.Conv2d(4,4,6, padding=0)
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(4, 64)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # define forward behavior
        
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        return x