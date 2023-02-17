from dataset import ZDataset
from torch.utils.data import DataLoader
from trainnvalid import train_valid, test
import numpy as np
from chess_cnn import Net, PlainChessNET

model = Net()


Dataset = "path/chess-game" # ZDataset(we_2000['AN'])
training_set = Dataset + "/train"
valid_set = Dataset + "/valid"
test_set = Dataset + "/test"


"""
LOADING SECTION
training_set = ZDataset(dataset['AN'])
"""


# normalization + convert to tensor
trainloader = DataLoader(training_set, batch_size=32, drop_last=True)
validloader = DataLoader(valid_set, batch_size=32, drop_last=True)
testloader = DataLoader(test_set, batch_size=32, drop_last=True)
 

"""from piece class variable legal moves are in self.ok_moves = []"""

def directmove_checkmate(board):
    board = board.copy() # see also simulation in board.py
    legal_moves = list(board.legal_moves)

    for move in legal_moves:
        board.push_uci(str(move))

        if board.is_checkmate():
            move = board.pop()
            return move
        _ = board.pop()

def distribution_moves(vals):
    probs = np.array(vals)
    probs = np.exp(probs)
    # normalization
    probs = probs / probs.sum()
    # increase the gap within distribution with power of 3
    probs = probs ** 3
    # normalization
    probs = probs / probs.sum()
    return probs