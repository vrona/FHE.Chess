from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from dataset_v3_target import Chessset
from pred_chess import predict


game_move_set = "/Volumes/vrona_SSD/lichess_data/wb_2000_300.csv"
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

#datafromset = Chessset(wechess['AN'])
trainset = Chessset(train['AN'], train.shape[0]) # 530025
validset = Chessset(valid['AN'], valid.shape[0]) # 176675
testset = Chessset(test['AN'], test.shape[0])    # 176676

train_data = DataLoader(trainset, batch_size = 1, shuffle=True, drop_last=True)
valid_data = DataLoader(validset, batch_size = 1, shuffle=True, drop_last=True)
test_data = DataLoader(testset, batch_size = 1, shuffle=True, drop_last=True)

c,s,t = next(iter(test_data))
print(s,t)