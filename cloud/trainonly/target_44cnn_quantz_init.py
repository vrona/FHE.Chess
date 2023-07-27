import brevitas.nn as qnn
#from brevitas.quant import Int8Bias
import torch
import torch.nn as nn
from torch.nn.utils import prune
import torch.nn.functional as F
import numpy as np


class QTtrgChessNET(nn.Module):

    def __init__(self, n_bits=6, w_bits=8, hidden_size=128):

        super(QTtrgChessNET, self).__init__()
        
        # define layers of CNN

        # input >> chessboard (12,8,8)

        self.quant_1 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.qinp_chessboard = qnn.QuantConv2d(12, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=w_bits)
        self.qbatchn1 = nn.BatchNorm2d(hidden_size) #qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu1 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False)

        self.quant_2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.qconv2 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=w_bits)
        self.qbatchn2 = nn.BatchNorm2d(hidden_size) #qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu2 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False)

        self.quant_3 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.qconv3 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=w_bits)
        self.qbatchn3 = nn.BatchNorm2d(hidden_size) #qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu3 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False)
        
        ## TEST
        self.quant_4 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.qconv4 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=w_bits)
        self.qbatchn4 = nn.BatchNorm2d(hidden_size) #qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu4 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False)
        
        self.quant_5 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.qconv5 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, weight_bit_width=w_bits)
        self.qbatchn5 = nn.BatchNorm2d(hidden_size) #qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu5 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False)
        ## TEST

        self.flatten = nn.Flatten()

        self.qfc1 = qnn.QuantLinear(hidden_size * 64, 64, bias=True, weight_bit_width=w_bits)
        self.q_chess_relu2 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False)

        # input >> source (the selected squares (64,) array)
        self.quant_inp_source = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.qinput_source = qnn.QuantLinear(64, 64, bias=True, weight_bit_width=w_bits)

        #self.quant_inp_merge = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        #self.qinput_merge = qnn.QuantLinear(64, 64, bias=True, weight_bit_width=w_bits)

        self.qbatchn1d_merge = nn.BatchNorm1d(64) #qnn.BatchNorm1dToQuantScaleBias(64, return_quant_tensor=False) # 

        # output target (the targeted square)
        self.qoutput_target = qnn.QuantLinear(64, 64, bias=True, weight_bit_width=w_bits)

        #self.qlast_relu = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)
        self.qSigmoid = qnn.QuantSigmoid(bit_width=n_bits, return_quant_tensor=False)

        # enable pruning
        self.pruning_conv(True)


    # from https://github.com/zama-ai/concrete-ml/blob/release/0.6.chessboard/docs/advanced_examples/ConvolutionalNeuralNetwork.ipynb
    def pruning_conv(self, enable):
            """Enables or removes pruning."""

            # Maximum number of active neurons (i.e. corresponding weight != 0)
            n_active = 64

            # Go through all the convolution layers
            for layer in (self.qinp_chessboard, self.qconv2, self.qconv3, self.qconv4, self.qconv5):
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
        
        chessboard = self.quant_4(chessboard)
        chessboard = self.qconv4(chessboard)
        chessboard = self.qbatchn4(chessboard)
        chessboard = self.qrelu4(chessboard)

        chessboard = self.quant_5(chessboard)
        chessboard = self.qconv5(chessboard)
        chessboard = self.qbatchn5(chessboard)
        chessboard = self.qrelu5(chessboard)

        chessboard = self.flatten(chessboard)

        chessboard = self.qfc1(chessboard)
        chessboard = self.q_chess_relu2(chessboard)

        source = self.quant_inp_source(source)
        source = self.qinput_source(source)

        # merging chessboard (context + selected source square)
        merge = chessboard + source
        #merge = torch.add(chessboard,source)
        #print("\nAVANT\n",merge)
        #merge = self.quant_inp_merge(merge)
        #merge = self.qinput_merge(merge)
        merge = self.qbatchn1d_merge(merge)
        #print("\nAPRES\n",merge)
        
        

        #merge = self.qrelu4(merge)
        #x = self.qSigmoid(merge)

        #x_target = self.qoutput_target(x)
        x_target = self.qSigmoid(self.qoutput_target(merge))

        return x_target
