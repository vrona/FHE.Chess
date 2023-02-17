from torch.utils.data import Dataset, Subset
import numpy as np
import chess
import pandas as pd
from matrices import Board_State, Move_State
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



we2000 = "/Volumes/vrona_SSD/lichess_data/we_2000_game_move.csv"
wechess = pd.read_csv(we2000)

class ZDataset(Dataset):
    
    def __init__(self, games):
        """which raw data"""
        super(ZDataset, self).__init__()
        self.games = games

        # self.evaluation = evaluation
        
        # # convert data to normalized floatTensor
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        #     ])
        
        # if train_on_gpu:
        #     self.games = torch.tensor(self.games, dtype=torch.float16).to("cuda")
        #     self.evaluation = torch.tensor(self.evaluation, dtype=torch.float16).to("cuda")
    def __repr__(self) -> str:
        return super(ZDataset, self).__repr__()

    def __len__(self):
        """returns the number of training data in batches"""
        return 40000


    def __getitem__(self, idx):

        game_i = np.random.randint(self.games.shape[0])
        random_game = we2000['AN'].values[game_i]
 
        moves = Move_State.list_move_sequence(random_game)

        game_state_i = np.random.randint(len(moves)-1)
        next_move = moves[game_state_i]
        moves = moves[:game_state_i]
        board = chess.Board()

        for move in moves:
            board.push_san(move)
            
            x = Board_State.board_tensor(board)
            y = Move_State.move_piece(next_move, board)

            # determine white or black turn (1 for w, -1 for b)
            if game_state_i %2 == 1:
                x *= -1
            print(x, y)
            return x, y
   
