import brevitas.nn as qnn
from brevitas.quant import Int32Bias
import torch
import torch.nn as nn
from torch.nn.utils import prune
import torch.nn.functional as F
import numpy as np

#N_FEAT = 12
n_bits = 8


class QTChessNET(nn.Module):

    def __init__(self, hidden_size=128):

        super(QTChessNET, self).__init__()
        

        # define layers of CNN

        # input >> chessboard 12*8*8
        self.quant_1 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.qinput_layer = qnn.QuantConv2d(12, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)
        self.qbatchn1 = qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu1 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

        self.quant_2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.qconv2 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)
        self.qbatchn2 = qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu2 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

        self.quant_3 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.qconv3 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)
        self.qbatchn3 = qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu3 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

        self.flatten = nn.Flatten()

        self.qfc1 = qnn.QuantLinear(hidden_size * 64, 64, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)
        self.qrelu2 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

        self.qoutput_source = qnn.QuantLinear(64, 64, bias=True, weight_bit_width=n_bits, bias_quant=Int32Bias)

        #self.qlast_relu = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)
        self.qSigmoid = qnn.QuantSigmoid(bit_width=n_bits, return_quant_tensor=True)

        # enable pruning
        self.pruning_conv(True)

    # from https://github.com/zama-ai/concrete-ml/blob/release/0.6.x/docs/advanced_examples/ConvolutionalNeuralNetwork.ipynb
    def pruning_conv(self, enable):
            """Enables or removes pruning."""

            # Maximum number of active neurons (i.e. corresponding weight != 0)
            n_active = 12

            # Go through all the convolution layers
            for layer in (self.qinput_layer, self.qconv2, self.qconv3):
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


    def forward(self, x):
        # define forward behavior

        # add sequence of convolutional
        x = self.quant_1(x)

        x = self.qinput_layer(x)
        x = self.qbatchn1(x)
        x = self.qrelu1(x)

        x = self.quant_2(x)
        x = self.qconv2(x)
        x = self.qbatchn2(x)
        x = self.qrelu2(x)

        x = self.quant_3(x)
        x = self.qconv3(x)
        x = self.qbatchn3(x)
        x = self.qrelu3(x)
        
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