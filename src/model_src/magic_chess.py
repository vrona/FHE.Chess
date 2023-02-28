from dataset import ZDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trainvalidtest import train_valid, test
import numpy as np
import pandas as pd
from chess_cnn import PlainChessNET
import wandb

# Dataset = "path/chess-game" # ZDataset(we_2000['AN'])
# training_set = Dataset + "/train"
# valid_set = Dataset + "/valid"
# test_set = Dataset + "/test"

"""
LOADING SECTION
training_set = ZDataset(dataset['AN'])
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


game_move_set = "/Volumes/vrona_SSD/lichess_data/we_2000_game_move.csv"
wechess = pd.read_csv(game_move_set)

# split dataset splitted into: training_set (80%), valid_set (20%), test_set (20%)
train, valid, test = np.split(wechess.sample(frac=1, random_state=42), [int(.6*len(wechess)), int(.8*len(wechess))])


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

#datafromset = ZDataset(wechess['AN'])
trainset = ZDataset(train['AN'], train.shape[0]) # 530025
validset = ZDataset(valid['AN'], valid.shape[0]) # 176675
testset = ZDataset(test['AN'], test.shape[0])   # 176676

train_loader = DataLoader(trainset, batch_size = 256, shuffle=True, drop_last=True)
valid_loader = DataLoader(validset, batch_size = 256, shuffle=True, drop_last=True)
test_loader = DataLoader(testset, batch_size = 256, shuffle=True, drop_last=True)

# Weights & Biases run tracking
api = wandb.Api()


run = api.run("vrona/Chess App/run")
run.config["learning_rate"] = 0.02
run.config["epochs"] = 2
run.update()

#model
model = PlainChessNET()
#loss
criterion = nn.CrossEntropyLoss()
#optimizer



train_valid(model, train_loader, valid_loader, criterion)

#model with lowest validation loss
model.loard_state_dict(torch.load("model_plain_chess.pt"))

test(model, test_loader, criterion)