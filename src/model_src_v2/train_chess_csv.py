import torch.nn as nn
from torch.utils.data import DataLoader
from train_v2 import train_valid, test
import numpy as np
import pandas as pd
from dataset_v3 import Chessset
from cnn_v10 import PlainChessNET


# Dataset = "path/chess-game" # Chessset(we_2000['AN'])
# training_set = Dataset + "/training_set"
# valid_set = Dataset + "/valid_set"
# test_set = Dataset + "/test"

"""
LOADING SECTION
training_set = Chessset(dataset['AN'])
"""
#       ___           ___           ___                   ___                    ___           ___           ___           ___     
#      /\  \         /\  \         /\__\      ___        /\  \                  /\  \         /\  \         /\  \         /\  \    
#     /::\  \       /::\  \       /:/  /     /\  \       \:\  \                /::\  \       /::\  \        \:\  \       /::\  \   
#    /:/\ \  \     /:/\:\  \     /:/  /      \:\  \       \:\  \              /:/\:\  \     /:/\:\  \        \:\  \     /:/\:\  \  
#   _\:\~\ \  \   /::\~\:\  \   /:/  /       /::\__\      /::\  \            /:/  \:\__\   /::\~\:\  \       /::\  \   /::\~\:\  \ 
#  /\ \:\ \ \__\ /:/\:\ \:\__\ /:/__/     __/:/\/__/     /:/\:\__\          /:/__/ \:|__| /:/\:\ \:\__\     /:/\:\__\ /:/\:\ \:\__\
#  \:\ \:\ \/__/ \/__\:\/:/  / \:\  \    /\/:/  /       /:/  \/__/          \:\  \ /:/  / \/__\:\/:/  /    /:/  \/__/ \/__\:\/:/  /
#   \:\ \:\__\        \::/  /   \:\  \   \::/__/       /:/  /                \:\  /:/  /       \::/  /    /:/  /           \::/  / 
#    \:\/:/  /         \/__/     \:\  \   \:\__\       \/__/                  \:\/:/  /        /:/  /     \/__/            /:/  /  
#     \::/  /                     \:\__\   \/__/                               \::/__/        /:/  /                      /:/  /   
#      \/__/                       \/__/                                        ~~            \/__/                       \/__/    


game_move_set = "/Volumes/vrona_SSD/lichess_data/wb_2000.csv"
wechess = pd.read_csv(game_move_set)

# split dataset splitted into: training_set (80%), valid_set (20%), test_set (20%)
training_set, valid_set, test_set = np.split(wechess.sample(frac=1, random_state=42), [int(.6*len(wechess)), int(.8*len(wechess))])

#      ___           ___           ___           ___           ___       ___           ___           ___     
#     /\  \         /\  \         /\  \         /\  \         /\__\     /\  \         /\  \         /\  \    
#    /::\  \       /::\  \        \:\  \       /::\  \       /:/  /    /::\  \       /::\  \       /::\  \   
#   /:/\:\  \     /:/\:\  \        \:\  \     /:/\:\  \     /:/  /    /:/\:\  \     /:/\:\  \     /:/\:\  \  
#  /:/  \:\__\   /::\~\:\  \       /::\  \   /::\~\:\  \   /:/  /    /:/  \:\  \   /::\~\:\  \   /:/  \:\__\ 
# /:/__/ \:|__| /:/\:\ \:\__\     /:/\:\__\ /:/\:\ \:\__\ /:/__/    /:/__/ \:\__\ /:/\:\ \:\__\ /:/__/ \:|__|
# \:\  \ /:/  / \/__\:\/:/  /    /:/  \/__/ \/__\:\/:/  / \:\  \    \:\  \ /:/  / \/__\:\/:/  / \:\  \ /:/  /
#  \:\  /:/  /       \::/  /    /:/  /           \::/  /   \:\  \    \:\  /:/  /       \::/  /   \:\  /:/  / 
#   \:\/:/  /        /:/  /     \/__/            /:/  /     \:\  \    \:\/:/  /        /:/  /     \:\/:/  /  
#    \::/__/        /:/  /                      /:/  /       \:\__\    \::/  /        /:/  /       \::/__/   
#     ~~            \/__/                       \/__/         \/__/     \/__/         \/__/         ~~       

#datafromset = Chessset(wechess['AN'])
trainset = Chessset(training_set['AN'], training_set.shape[0])  # 530025
validset = Chessset(valid_set['AN'], valid_set.shape[0])        # 176675
testset = Chessset(test_set['AN'], test_set.shape[0])           # 176676

train_loader = DataLoader(trainset, batch_size = 64, shuffle=True, drop_last=True)
valid_loader = DataLoader(validset, batch_size = 64, shuffle=True, drop_last=True)
test_loader = DataLoader(testset, batch_size = 64, shuffle=True, drop_last=True)


#model
model = PlainChessNET()
#loss
#criterion = nn.CrossEntropyLoss() #reduction='sum'
criterion_f = nn.MSELoss()
criterion_t = nn.MSELoss()
#criterion = nn.L1Loss()
#criterion = nn.NLLLoss()
#optimizer

train_valid(model, train_loader, valid_loader, criterion_f, criterion_t)

### model with lowest validation loss
#model.load_state_dict(torch.load("model_plain_chess.pt"))

# Test and accuracy
#test(model, test_loader, criterion)