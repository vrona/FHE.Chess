from torch.utils.data import Dataset, Subset
import numpy as np
import chess
from peewee import *
from helper_chess_v4 import Board_State, Move_State


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


# db = SqliteDatabase("/Volumes/vrona_SSD/lichess_data/chess_wb2000_db.db")


class Games(Model):
    id = IntegerField()
    AN = TextField()



helper_board_state = Board_State() 
helper_move_state = Move_State() 


class Chessset(Dataset):
    
    
    def __init__(self, games_df, size_data_set):
        """which raw data"""
        super(Chessset, self).__init__()
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
        
        # get the game from sql database or from csv file
        random_game = Games.get(Games.id == idx) # sql
        print(random_game.AN)
        # random_game = self.games_df.values[idx] # csv
        moves_sequence = helper_move_state.list_move_sequence(random_game.AN)

        # get random move
        game_state_i = np.random.randint(len(moves_sequence)-1)
        next_move = moves_sequence[game_state_i]  # piece_pos

        # get the sequence of moves until the move
        moves = moves_sequence[:game_state_i]

        # get the eval for the sequence until that move

        board = chess.Board()

        for move in moves:
            board.push_san(move)

        x = helper_board_state.board_tensor(board)          # shape(6,8,8)

        #y = helper_move_state.move_piece(next_move, board)  # shape(2,8,8)

        #t = helper_move_state.test_from(next_move, board)
        
        z = helper_move_state.choose_piece(next_move, board) # shape (1)

        # get the eval
        # xx = get_eval() of piece_pos
        # yy = get_eval() of move_pos

        # determine white or black turn (1 for w, -1 for b) and then the one to play has always positive value
        if game_state_i %2 == 1:
            x *= -1
        return x, z