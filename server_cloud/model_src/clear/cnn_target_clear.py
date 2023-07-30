import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, hidden_size):

        super(Net, self).__init__()
        # define layers of CNN

        # input >> hidden layer
        self.conv1 = nn.Conv2d(hidden_size, hidden_size,kernel_size=3, stride=1, padding=1)
        self.batchn1 = nn.BatchNorm2d(hidden_size)

        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.batchn2 = nn.BatchNorm2d(hidden_size)

    def forward(self, chessboard):
        # define forward behavior

        # activations and batch normalization
        chessboard = self.conv1(chessboard)
        chessboard = self.batchn1(chessboard)
        chessboard = F.relu(chessboard)

        chessboard = self.conv2(chessboard)
        chessboard = self.batchn2(chessboard)
        chessboard = F.relu(chessboard)

        return chessboard


class PlainChessNET(nn.Module):

    def __init__(self, hidden_layers=2, hidden_size=128):

        super(PlainChessNET, self).__init__()
        
        # define layers of CNN

        # chessboard part (12,8,8) input >> hidden layer
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(12, hidden_size, kernel_size=3, stride=1, padding=1)
        self.modulelist = nn.ModuleList([Net(hidden_size) for i in range(hidden_layers)])
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_size * 64, 64)

        # source (the selected squares)
        self.input_source = nn.Linear(64,64)

        self.batchn1d_1 = nn.BatchNorm1d(64)
        
        # output target (the targeted square)
        self.output_target = nn.Linear(64,64)


    def forward(self, chessboard, source):
        # define forward behavior

        # chessboard context
        chessboard = self.input_layer(chessboard)
        chessboard = F.relu(chessboard)

        for h in range(self.hidden_layers):
            chessboard = self.modulelist[h](chessboard)
        

        chessboard = self.flatten(chessboard)

        chessboard = self.fc1(chessboard)
        chessboard = F.relu(chessboard)

        # source aka selected square to be moved to target
        source = self.input_source(source)

        # merging chessboard (context + selected source square)
        merge = chessboard + source
        merge = self.batchn1d_1(merge)

        # nllloss crossentropyloss
        # x_target = F.log_softmax(self.output_target(merge),dim=1)

        # mseloss
        #x_target = F.relu(self.output_target(merge))

        # sigmoid
        x_target = torch.sigmoid(self.output_target(merge))

        return x_target