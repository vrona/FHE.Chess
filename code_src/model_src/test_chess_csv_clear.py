import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from dataset_v3_source import Chessset
from train_v3_source_clear import train_valid, test
from cnn_v13_64bit_source_clear import PlainChessNET
# from dataset_v3_target import Chessset
# from train_v3_target import train_valid, test
# from cnn_v13_64bit_target_unfhe import PlainChessNET

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


game_move_set = "/Volumes/vrona_SSD/lichess_data/wb_2000_300.csv"
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
trainset = Chessset(training_set['AN'], training_set.shape[0])
validset = Chessset(valid_set['AN'], valid_set.shape[0])
testset = Chessset(test_set['AN'], test_set.shape[0])

train_loader = DataLoader(trainset, batch_size = 64, shuffle=True, drop_last=True)
valid_loader = DataLoader(validset, batch_size = 64, shuffle=True, drop_last=True)
test_loader = DataLoader(testset, batch_size = 1, shuffle=True, drop_last=True)

#model
model = PlainChessNET()

#loss
criterion = nn.MSELoss()

## TRAINING
#train_valid(model, train_loader, valid_loader, criterion, criterion)

## TESTING
model.load_state_dict(torch.load("server/model/source_clear.pt")) #source
#model.load_state_dict(torch.load("server/model/target_clear.pt")) #target

# Test and accuracy
test(model, test_loader, criterion)