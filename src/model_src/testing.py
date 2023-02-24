from dataset import ZDataset
from dataset import train_val_splitter
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np



game_move_set = "/Volumes/vrona_SSD/lichess_data/we_2000_game_move.csv"
wechess = pd.read_csv(game_move_set)

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
trainset = ZDataset(train['AN'], 530025) # 530025
validset = ZDataset(valid['AN'], 176675) # 176675
testset = ZDataset(test['AN'], 176675)   # 176676

train_data = DataLoader(trainset, batch_size = 40000, shuffle=True, drop_last=True)
valid_data = DataLoader(validset, batch_size = 40000, shuffle=True, drop_last=True)
test_data = DataLoader(testset, batch_size = 40000, shuffle=True, drop_last=True)

x, y = next(iter(valid_data))
print(x, y)
#first = datafromset[0]
#print(datafromset)
# x, y = first
# print(x, y)

