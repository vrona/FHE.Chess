import brevitas.nn as qnn
import torch.nn as nn
from torch.nn.utils import prune
import numpy as np


class QTtrgChessNET(nn.Module):

    def __init__(self, n_bits=4, w_bits=4, rqt=True, b_q = None, hidden_size=128):

        super(QTtrgChessNET, self).__init__()
        
        # define layers of CNN

        # input >> chessboard (12,8,8)

        self.quant_1 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=rqt)
        self.qinp_chessboard = qnn.QuantConv2d(12, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, bias_quant=b_q, weight_bit_width=w_bits)
        #self.qbatchn1 = qnn.BatchNorm2dToQuantScaleBias(hidden_size) #nn.BatchNorm2d(hidden_size)
        self.qrelu1 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=rqt)
        
        self.quant_2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=rqt)
        self.qconv2 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, bias_quant=b_q, weight_bit_width=w_bits)
        #self.qbatchn2 = qnn.BatchNorm2dToQuantScaleBias(hidden_size) #nn.BatchNorm2d(hidden_size) #qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu2 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=rqt)

        self.quant_3 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=rqt)
        self.qconv3 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, bias_quant=b_q, weight_bit_width=w_bits)
        #self.qbatchn3 = qnn.BatchNorm2dToQuantScaleBias(hidden_size) #nn.BatchNorm2d(hidden_size) #qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu3 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=rqt)
        
        self.quant_4 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=rqt)
        self.qconv4 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, bias_quant=b_q, weight_bit_width=w_bits)
        #self.qbatchn4 = qnn.BatchNorm2dToQuantScaleBias(hidden_size) #nn.BatchNorm2d(hidden_size) #qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu4 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=rqt)
        
        self.quant_5 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=rqt)
        self.qconv5 = qnn.QuantConv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, bias_quant=b_q, weight_bit_width=w_bits)
        #self.qbatchn5 = qnn.BatchNorm2dToQuantScaleBias(hidden_size) #nn.BatchNorm2d(hidden_size) #qnn.BatchNorm2dToQuantScaleBias(hidden_size)
        self.qrelu5 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=rqt)
        
        self.flatten = nn.Flatten()

        self.qfc1 = qnn.QuantLinear(hidden_size * 64, 64, bias=True, bias_quant=b_q, weight_bit_width=w_bits)
        #self.q_flat_chess_relu = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=rqt)

        # input >> source (the selected squares (64,) array)
        #self.quant_source1 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=rqt)
        self.qinput_source = qnn.QuantLinear(64, 64, bias=True, bias_quant=b_q, weight_bit_width=w_bits)
        #self.q_source_relu1 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=rqt)
        
        self.qbatchn1d_merge = nn.BatchNorm1d(64)

        self.quant_merge = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=rqt)
        self.q_merge_relu = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=rqt)
        
        self.qSigmoid = qnn.QuantSigmoid(bit_width=n_bits, return_quant_tensor=False)
        
        # output target (the targeted square)
        self.qoutput_target = qnn.QuantLinear(64, 64, bias=True, bias_quant=b_q, weight_bit_width=w_bits)
        
        # enable pruning
        self.pruning_conv(True)


    # from https://github.com/zama-ai/concrete-ml/blob/release/0.6.chessboard/docs/advanced_examples/ConvolutionalNeuralNetwork.ipynb
    def pruning_conv(self, enable):
            """Enables or removes pruning."""

            # Maximum number of active neurons (i.e. corresponding weight != 0)
            n_active = 84

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
        #chessboard = self.qbatchn1(chessboard)
        chessboard = self.qrelu1(chessboard)

        chessboard = self.quant_2(chessboard)
        chessboard = self.qconv2(chessboard)
        #chessboard = self.qbatchn2(chessboard)
        chessboard = self.qrelu2(chessboard)

        chessboard = self.quant_3(chessboard)
        chessboard = self.qconv3(chessboard)
        #chessboard = self.qbatchn3(chessboard)
        chessboard = self.qrelu3(chessboard)
        
        chessboard = self.quant_4(chessboard)
        chessboard = self.qconv4(chessboard)
        #chessboard = self.qbatchn4(chessboard)
        chessboard = self.qrelu4(chessboard)
        
        chessboard = self.quant_5(chessboard)
        chessboard = self.qconv5(chessboard)
        #chessboard = self.qbatchn5(chessboard)
        chessboard = self.qrelu5(chessboard)
        
        chessboard = self.flatten(chessboard)
        
        chessboard = self.qfc1(chessboard)
        #chessboard = self.q_flat_chess_relu(chessboard)

        #source = self.quant_source1(source)
        source = self.qinput_source(source)
        #source = self.q_source_relu1(source)
    
        # merging chessboard (context + selected source square)
        self.quant_merge.eval()
        chessboard_eval = self.quant_merge(chessboard)
        source_eval = self.quant_merge(source)

        #print("SCALE -->",chessboard.scale.item()-source.scale.item())
        merge = chessboard_eval + source_eval
        merge = self.qbatchn1d_merge(merge)
        merge = self.q_merge_relu(merge)     

        ## not good
        #x = self.qSigmoid(merge)
        #x_target = self.qoutput_target(x)
        
        ## better
        x = self.qoutput_target(merge)
        x_target = self.qSigmoid(x)

        return x_target
