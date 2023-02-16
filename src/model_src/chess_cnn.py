import torch
import torch.nn as nn
import torch.nn.functional as F

"""
w: input volume size
F: kernel/filter size
P: amount of zero padding
S: stride
output_w : (W−F+2P)/S+1
eg.: if input 7*7 / filter 3*3 / s 1 / pad 0 then output_w 3*3
"""

class FHEChessNet(nn.Module):

    def __init__(self, hidden_layers_):
        super(FHEChessNet, self).__init__()
        # define layers of CNN

        # input >> hidden layer
        self.conv1 = nn.Conv2d(6,2,kernel_size=3, stride=1, padding=1)
        self.batchn1 = nn.BatchNorm2d()

        self.conv2 = nn.Conv2d(4,2, kernel_size=3, stride=1, padding=1)
        self.batchn2 = nn.BatchNorm2d()

    def forward(self, x):
        # define forward behavior
        
        x = x.view(-1, 1 * 8 * 8)
        # add sequence of convolutional
        #x = F.max_pool2d()
        x = F.selu(self.conv1(x))
        x = self.batchn1(x)

        x = F.selu(self.conv2(x))
        x = self.batchn2(x)
        # flatten
        x = x.view(-1, x.size(1))
        # full connect
        #x = F.linear()
        # softmax activation

        
        return x
    

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

        # add sequence of convolutional
        #x = F.max_pool2d()
        x = F.selu(self.conv1(x))
        x = self.batchn1(x)

        x = F.selu(self.conv2(x))
        x = self.batchn2(x)

        x = x + x_input
        return x


class PieceNet(nn.Module):

    def __init__(self, hidden_layers=4, hidden_size=200):

        super(PieceNet, self).__init__()
        # define layers of CNN

        # input >> hidden layer
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size,kernel_size=3, stride=1, padding=1)
        self.modulelist = nn.ModuleList([nn.Linear(hidden_size) for i in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # define forward behavior

        # add sequence of convolutional
        #x = F.max_pool2d()
        x = self.input_layer(x)
        x = F.relu(x)

        for h in range(self.hidden_layers):
            x = self.modulelist[h](x)
        
        x = self.output_layer(x)

        return x
    
"""best probability distribution from each output feature map * conditioned to rules:
- only legal move (ban illegal)
- randomness (check stochastic)
- adhoc function
- input feature maps
"""