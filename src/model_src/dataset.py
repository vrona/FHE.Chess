from torch.utils.data import Dataset, Subset
import numpy as np
import chess
import pandas as pd
from helper_chess import Board_State, Move_State
from sklearn.model_selection import train_test_split


#       ___           ___           ___           ___           ___           ___           ___     
#      /\  \         /\  \         /\  \         /\  \         /\  \         /\  \         /\  \    
#     /::\  \       /::\  \        \:\  \       /::\  \       /::\  \       /::\  \        \:\  \   
#    /:/\:\  \     /:/\:\  \        \:\  \     /:/\:\  \     /:/\ \  \     /:/\:\  \        \:\  \  
#   /:/  \:\__\   /::\~\:\  \       /::\  \   /::\~\:\  \   _\:\~\ \  \   /::\~\:\  \       /::\  \ 
#  /:/__/ \:|__| /:/\:\ \:\__\     /:/\:\__\ /:/\:\ \:\__\ /\ \:\ \ \__\ /:/\:\ \:\__\     /:/\:\__\
#  \:\  \ /:/  / \/__\:\/:/  /    /:/  \/__/ \/__\:\/:/  / \:\ \:\ \/__/ \:\~\:\ \/__/    /:/  \/__/
#   \:\  /:/  /       \::/  /    /:/  /           \::/  /   \:\ \:\__\    \:\ \:\__\     /:/  /     
#    \:\/:/  /        /:/  /     \/__/            /:/  /     \:\/:/  /     \:\ \/__/     \/__/      
#     \::/__/        /:/  /                      /:/  /       \::/  /       \:\__\                  
#      ~~            \/__/                       \/__/         \/__/         \/__/                


def train_val_splitter(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    print(len(datasets['train']))
    datasets['val'] = Subset(dataset, val_idx)
    print(len(datasets['val']))

    return datasets

helper_board_state = Board_State() 
helper_move_state = Move_State() 

we2000 = "/Volumes/vrona_SSD/lichess_data/we_2000_game_move.csv"
wechess = pd.read_csv(we2000)

class ZDataset(Dataset):
    
    def __init__(self, games_df, size_data_set):
        """which raw data"""
        super(ZDataset, self).__init__()
        self.size_data_set = size_data_set
        self.games_df = games_df
        
        # self.evaluation = evaluation
        
        # # convert data to normalized floatTensor
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        #     ])
        
        # if train_on_gpu:
        #     self.games_df = torch.tensor(self.games_df, dtype=torch.float16).to("cuda")
        #     self.evaluation = torch.tensor(self.evaluation, dtype=torch.float16).to("cuda")


    def __len__(self):
        """returns the dataset's size"""
        return self.size_data_set #total 883375


    def __getitem__(self, idx):

        # game_i = np.random.randint(self.games_df.shape[0])
        # random_game = we2000['AN'].values[game_i]
        random_game = self.games_df.values[idx] #
        initial_moves = helper_move_state.list_move_sequence(random_game)

        game_state_i = np.random.randint(len(initial_moves)-1)
        next_move = initial_moves[game_state_i]

        moves = initial_moves[:game_state_i]

        board = chess.Board()

        for move in moves:
            board.push_san(move)

        x = helper_board_state.board_tensor(board)
        y = helper_move_state.move_piece(next_move, board)

        # determine white or black turn (1 for w, -1 for b)
        if game_state_i %2 == 1:
            x *= -1
        return x, y
   