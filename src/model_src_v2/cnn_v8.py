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

class PlainChessNET(nn.Module):

    def __init__(self):

        super(PlainChessNET, self).__init__()
        
        # define layers of CNN

        # input >> hidden layer

        self.input_layer = nn.Conv2d(6, 384, kernel_size=3, stride=1, padding=1)
        self.batchn1 = nn.BatchNorm2d(384)


        self.conv2 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.batchn2 = nn.BatchNorm2d(384)

        self.conv3 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.batchn3 = nn.BatchNorm2d(384)

        self.conv5 = nn.Conv2d(384, 64, kernel_size=3, stride=1, padding=1)
        self.batchn5 = nn.BatchNorm2d(64)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(4096, 64)
        


    def convo(self, x):
        
        x = self.input_layer(x)

        x = self.batchn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.batchn2(x)
        x = F.selu(x)

        x = self.conv3(x)
        x = self.batchn3(x)
        x = F.selu(x)

        x = self.conv5(x)
        x = self.batchn5(x)
        x = F.selu(x)
       

        return x
    
    def forward(self, x):
        # define forward behavior
        #x_input = torch.clone(x)
        #x = x + x_input # residual
        
        # add sequence of convolutional
        x = self.convo(x)
        
        #x = self.output(x) # torch.Size([64, 8])
        #x = F.softmax(x, dim=1)

        x = self.flatten(x)
        x = self.fc1(x)


        return x
    
"""best probability distribution from each output feature map * conditioned to rules:
- only legal move (ban illegal)
- randomness (check stochastic)
- adhoc function
- input feature maps
"""

# class FHEChessNet(nn.Module):

#     def __init__(self):
#         super(FHEChessNet, self).__init__()
#         # define layers of CNN

#         # input >> hidden layer # shape 8x8x6
#         self.conv1 = nn.Conv2d(6, 8, kernel_size=3, stride=1, padding=1)
#         self.batchn1 = nn.BatchNorm2d()

#         self.conv2 = nn.Conv2d(4,2, kernel_size=3, stride=1, padding=1)
#         self.batchn2 = nn.BatchNorm2d()

#         self.flatten = nn.Flatten()
#     def forward(self, x):
#         # define forward behavior
        
#         x = x.view(-1, 1 * 8 * 8)
#         # add sequence of convolutional
#         #x = F.max_pool2d()
#         x = F.relu(self.conv1(x))
#         x = self.batchn1(x)

#         x = F.selu(self.conv2(x))
#         x = self.batchn2(x)
#         # flatten
#         x = self
#         x = x.view(-1, x.size(1))
#         # full connect
#         #x = F.linear()
#         # softmax activation

        
#         return x