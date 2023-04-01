############--ABORDED--############--ABORDED--############
from dataset_v3 import Chessset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_v2 import train_valid, test
import numpy as np
import pandas as pd
from cnn_v6 import PlainChessNET
from peewee import *

# Dataset = "path/chess-game" # Chessset(we_2000['AN'])
# training_set = Dataset + "/training_set"
# valid_set = Dataset + "/valid_set"
# test_set = Dataset + "/test"

"""
LOADING SECTION
training_set = Chessset(dataset['AN'])
"""

db_train = SqliteDatabase("/Volumes/vrona_SSD/lichess_data/chess_wb2000_train.db")
db_valid = SqliteDatabase("/Volumes/vrona_SSD/lichess_data/chess_wb2000_valid.db")
db_test = SqliteDatabase("/Volumes/vrona_SSD/lichess_data/chess_wb2000_test.db")


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

# ************ NOT WORKING ************ NOT WORKING ************
# class Games(Model):
#     id = IntegerField()
#     AN = TextField()

#     class Meta:
#         database = db_train

# db_train.connect()


#datafromset = Chessset(wechess['AN'])
trainset = Chessset(db_train, 327868)        # 530025
validset = Chessset(db_valid, 109289)        # 176675
testset = Chessset(db_test, 109290)           # 176676

train_loader = DataLoader(trainset, batch_size = 64, shuffle=True, drop_last=True)
valid_loader = DataLoader(validset, batch_size = 64, shuffle=True, drop_last=True)
test_loader = DataLoader(testset, batch_size = 64, shuffle=True, drop_last=True)


#model
model = PlainChessNET()
#loss
criterion = nn.CrossEntropyLoss(reduction='sum')
#criterion = nn.MSELoss()
#optimizer

train_valid(model, train_loader, valid_loader, criterion)

### model with lowest validation loss
#model.load_state_dict(torch.load("model_plain_chess.pt"))

# Test and accuracy
#test(model, test_loader, criterion)

db.close()