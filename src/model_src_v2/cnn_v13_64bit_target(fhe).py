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


    def forward(self, chessboard):
        # define forward behavior

        # activations and batch normalization
        chessboard = self.quant_inp(chessboard)
        chessboard = self.qconv1(chessboard)
        chessboard = self.qbatchn1(chessboard)
        chessboard = self.qrelu1(chessboard)

        chessboard = self.qconv2(chessboard)
        chessboard = self.qbatchn2(chessboard)
        chessboard = self.qrelu2(chessboard)

        return chessboard


class QTtrgChessNET(nn.Module):

    def __init__(self, hidden_layers=2, hidden_size=128):

        super(QTtrgChessNET, self).__init__()
        
        # define layers of CNN

        # chessboard part (12,8,8) input >> hidden layer
        self.hidden_layers = hidden_layers
        self.quant_inp_chessboard = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.qinput_layer = qnn.QuantConv2d(12, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)
        self.qrelu1 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

        self.qmodulelist = nn.ModuleList([QTNet(hidden_size) for i in range(hidden_layers)])

        self.flatten = nn.Flatten()

        self.qfc1 = qnn.QuantLinear(hidden_size * 64, 64, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)
        self.qrelu2 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

        # source (the selected squares)
        self.quant_inp_source = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.qinput_source = qnn.QuantLinear(64, 64, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)

        self.qbatchn1d_1 = qnn.BatchNorm1dToQuantScaleBias(64)
        
        # output target (the targeted square)
        self.qoutput_target = qnn.QuantLinear(64, 64, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)

        #self.qlast_relu = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)
        self.qSigmoid = qnn.QuantSigmoid(bit_width=n_bits, return_quant_tensor=True)


    def forward(self, chessboard, source):
        # define forward behavior

        # add sequence of convolutional
        chessboard = self.quant_inp_chessboard(chessboard)
        chessboard = self.qinput_layer(chessboard)
        chessboard = self.qrelu1(chessboard)

        for h in range(self.hidden_layers):
            chessboard = self.qmodulelist[h](chessboard)
        

        chessboard = self.flatten(chessboard)

        chessboard = self.qfc1(chessboard)
        chessboard = self.qrelu2(chessboard)
        
        source = self.quant_inp_source(source)
        source = self.qinput_source(source)

        # merging chessboard (context + selected source square)
        merge = chessboard + source
        merge = self.qbatchn1d_1(merge)

        # nllloss crossentropyloss
        # x_target = F.log_softmax(self.output_target(merge),dim=1)

        # mseloss
        #x_target = F.relu(self.output_target(merge))

        # sigmoid
        x_target = self.qSigmoid(self.qoutput_target(merge))

        return x_target
