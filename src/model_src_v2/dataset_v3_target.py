from torch.utils.data import Dataset
import numpy as np
import chess
from helper_chess_v7_64target import Board_State, Move_State


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

        # get the game
        random_game = self.games_df.values[idx]
        initial_moves = helper_move_state.list_move_sequence(random_game)

        # get random move
        game_state_i = np.random.randint(len(initial_moves)-1)
        next_move = initial_moves[game_state_i]  # piece_pos

        # get the sequence of moves until the move
        moves = initial_moves[:game_state_i]

        # get the eval for the sequence until that move

        board = chess.Board()

        for move in moves:
            board.push_san(move)

        c = helper_board_state.board_tensor_12(board)          # shape(6,8,8) or shape(12,8,8)
        
        s,t = helper_move_state.from_to_bitboards(next_move, board) # shape (1)

        # get the eval
        # xx = get_eval() of piece_pos
        # yy = get_eval() of move_pos

        # determine white or black turn (1 for w, -1 for b) and then the one to play has always positive value
        if game_state_i %2 == 1:
            c *= -1
        return c,s,t