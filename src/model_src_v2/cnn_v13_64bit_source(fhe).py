import brevitas.nn as qnn
from brevitas.quant import Int32Bias
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

#N_FEAT = 12
n_bits = 8

class QTNet(nn.Module):

    def __init__(self, hidden_size):

        super(QTNet, self).__init__()
        # define layers of CNN

        # input >> hidden layer
        self.quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.qconv1 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)
        #self.conv1 = nn.Conv2d(hidden_size, hidden_size,kernel_size=3, stride=1, padding=1)
        #self.batchn1 = nn.BatchNorm2d(hidden_size)


        self.qconv2 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias))
        #self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        #self.batchn2 = nn.BatchNorm2d(hidden_size)

    def forward(self, x):
        # define forward behavior
        #x_input = torch.clone(x)

        # activations and batch normalization
        x = self.quant_inp(x)
        x = self.qconv1(x)
        x = self.batchn1(x)
        x = F.relu(x)

        x = self.qconv2(x)
        x = self.batchn2(x)

        #x = x + x_input
        x = F.relu(x)

        return x


class QTChessNET(nn.Module):

    def __init__(self, hidden_layers=2, hidden_size=128):

        super(QTChessNET, self).__init__()
        
        # define layers of CNN

        # input >> hidden layer
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(12, hidden_size, kernel_size=3, stride=1, padding=1)
        self.modulelist = nn.ModuleList([QTNet(hidden_size) for i in range(hidden_layers)])
        #self.last_conv = nn.Conv2d(hidden_size, 64, kernel_size=3, stride=1, padding=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(hidden_size * 64, 64)
        self.batchn1d_1 = nn.BatchNorm1d(64)
        self.output_source = nn.Linear(64,64)


    def forward(self, x):
        # define forward behavior

        # add sequence of convolutional
        #x = F.max_pool2d()
        x = self.input_layer(x)
        x = F.relu(x)

        for h in range(self.hidden_layers):
            x = self.modulelist[h](x)
        
        #x = self.last_conv(x) # torch.Size([64, 128, 8, 8])
        x = self.flatten(x)

        x = self.fc1(x)
        x = F.relu(x)
        #x = self.batchn1d_1(x)

        # nllloss crossentropyloss
        # x_source = F.log_softmax(self.output_source(x),dim=1)

        # mseloss
        #x_source = F.relu(self.output_source(x))

        # sigmoid
        x_source = torch.sigmoid(self.output_source(x))

        return x_source