import brevitas.nn as qnn
from brevitas.quant import Int32Bias
import torch
import torch.nn as nn
import torch.nn.functional as F

#N_FEAT = 12
n_bits = 8

class QTNet(nn.Module):

    def __init__(self, hidden_size):

        super(QTNet, self).__init__()
        # define layers of CNN

        # input >> hidden layer
        self.quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.qconv1 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)
        self.qbatchn1 = qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu1 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

        self.qconv2 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)
        self.qbatchn2 = qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu2 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)


    def forward(self, x):
        # define forward behavior

        # activations and batch normalization
        x = self.quant_inp(x)
        x = self.qbatchn1(x)
        x = self.qrelu1(x)


        x = self.qconv2(x)
        x = self.qrelu2(x)

        return x


class QTsrcChessNET(nn.Module):

    def __init__(self, hidden_layers=2, hidden_size=128):

        super(QTsrcChessNET, self).__init__()
        
        # define layers of CNN

        # input >> hidden layer
        self.hidden_layers = hidden_layers
        self.quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.qinput_layer = qnn.QuantConv2d(12, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)
        self.qrelu1 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

        self.qmodulelist = nn.ModuleList([QTNet(hidden_size) for i in range(hidden_layers)])

        self.flatten = nn.Flatten()

        self.qfc1 = qnn.QuantLinear(hidden_size * 64, 64, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)
        self.qrelu2 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

        self.qoutput_source = qnn.QuantLinear(64, 64, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)

        #self.qlast_relu = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)
        self.qSigmoid = qnn.QuantSigmoid(bit_width=n_bits, return_quant_tensor=True)


    def forward(self, x):
        # define forward behavior

        # add sequence of convolutional
        x = self.quant_inp(x)
        x = self.qinput_layer(x)
        x = self.qrelu1(x)

        for h in range(self.hidden_layers):
            x = self.qmodulelist[h](x)
        
        # torch.Size([64, 128, 8, 8])
        x = self.flatten(x)

        x = self.qfc1(x)
        x = self.qrelu2(x)

        ## nllloss
        #x_source = torch.log_softmax(self.qoutput_source(x),dim=1)

        ## mseloss
            # relu
        #x_source = self.qlast_relu(self.qoutput_source(x))

            # sigmoid
        x_source = self.qSigmoid(self.qoutput_source(x))

        return x_source