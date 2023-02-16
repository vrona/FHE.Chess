import numpy as np
import chess
from matrices import Board_State, Move_State

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


    def __len__(self):
        """returns the number of training data in batches"""
        return 40000


    def __getitem__(self, Dataset): # index ? chess_data['AN']
        game_i = np.random.randint(self.games.shape[0])
        random_game = Dataset.values[game_i]
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
            return x, y
   
