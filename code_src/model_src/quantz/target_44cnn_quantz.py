import brevitas.nn as qnn
from brevitas.quant import Int32Bias
import torch
import torch.nn as nn
from torch.nn.utils import prune
import torch.nn.functional as F
import numpy as np


class QTtrgChessNET(nn.Module):

    def __init__(self, n_bits=4, w_bits=4, hidden_size=128):

        super(QTtrgChessNET, self).__init__()
        
        # define layers of CNN

        # input >> chessboard (12,8,8)

        self.quant_1 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.qinp_chessboard = qnn.QuantConv2d(12, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=w_bits)
        self.qbatchn1 = qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu1 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

        self.quant_2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.qconv2 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=w_bits)
        self.qbatchn2 = qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu2 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

        self.quant_3 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.qconv3 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=w_bits)
        self.qbatchn3 = qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu3 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

        self.flatten = nn.Flatten()

        self.qfc1 = qnn.QuantLinear(hidden_size * 64, 64, bias=True, weight_bit_width=w_bits)
        self.qrelu2 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

        # input >> source (the selected squares (64,) array)
        self.quant_inp_source = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.qinput_source = qnn.QuantLinear(64, 64, bias=True, weight_bit_width=w_bits)

        self.qbatchn1d_merge = nn.BatchNorm1d(64)#qnn.BatchNorm2dToQuantScaleBias(64)

        # output target (the targeted square)
        self.qoutput_target = qnn.QuantLinear(64, 64, bias=True, weight_bit_width=w_bits)

        #self.qlast_relu = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)
        self.qSigmoid = qnn.QuantSigmoid(bit_width=n_bits, return_quant_tensor=True)

        # enable pruning
        self.pruning_conv(True)


    # from https://github.com/zama-ai/concrete-ml/blob/release/0.6.chessboard/docs/advanced_examples/ConvolutionalNeuralNetwork.ipynb
    def pruning_conv(self, enable):
            """Enables or removes pruning."""

            # Maximum number of active neurons (i.e. corresponding weight != 0)
            n_active = 64

            # Go through all the convolution layers
            for layer in (self.qinp_chessboard, self.qconv2, self.qconv3):
                s = layer.weight.shape

                # Compute fan-in (number of inputs to a neuron)
                # and fan-out (number of neurons in the layer)
                st = [s[0], np.prod(s[1:])]

                # The number of input neurons (fan-in) is the product of
                # the kernel width x height x inChannels.
                if st[1] > n_active:
                    if enable:
                        # This will create a forward hook to create a mask tensor that is multiplied
                        # with the weights during forward. The mask will contain 0s or 1s
                        prune.l1_unstructured(layer, "weight", (st[1] - n_active) * st[0])
                    else:
                        # When disabling pruning, the mask is multiplied with the weights
                        # and the result is stored in the weights member
                        prune.remove(layer, "weight")



    def forward(self, chessboard, source):
        # define forward behavior

        # add sequence of convolutional
        chessboard = self.quant_1(chessboard)
        chessboard = self.qinp_chessboard(chessboard)
        chessboard = self.qbatchn1(chessboard)
        chessboard = self.qrelu1(chessboard)

        chessboard = self.quant_2(chessboard)
        chessboard = self.qconv2(chessboard)
        chessboard = self.qbatchn2(chessboard)
        chessboard = self.qrelu2(chessboard)

        chessboard = self.quant_3(chessboard)
        chessboard = self.qconv3(chessboard)
        chessboard = self.qbatchn3(chessboard)
        chessboard = self.qrelu3(chessboard)
        
        chessboard = self.flatten(chessboard)

        chessboard = self.qfc1(chessboard)
        chessboard = self.qrelu2(chessboard)
        
        source = self.quant_inp_source(source)
        source = self.qinput_source(source)

        # merging chessboard (context + selected source square)
        merge = chessboard + source
        merge = self.qbatchn1d_merge(merge)
        #print(merge.shape)

        #merge = self.qrelu4(merge)
        x = self.qSigmoid(merge)

        x_target = self.qoutput_target(x)
        #x_target = self.qSigmoid(self.qoutput_target(merge))

        return x_target