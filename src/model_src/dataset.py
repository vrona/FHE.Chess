import numpy as np
import chess
from matrices import Board_State, Move_State

class ZDataset(Dataset):
  
  def __init__(self, games):
    """which raw data"""
    super(ZDataset, self).__init__()
    self.games = games

  def __len__(self):
    """how much training data in batches"""
    return 40000


  def __getitem__(self, index):
    game_i = np.random.randint(self.games.shape[0])
    random_game = chess_data['AN'].values[game_i]
    moves = Move_State.list_move_sequence(random_game)

    game_state_i = np.random.randint(len(moves)-1)
    next_move = moves[game_state_i]
    moves = moves[:game_state_i]
    board = chess.Board()

    for move in moves:
      board.push_san(move)
    
    x = Board_State.board_2_rep(board)
    y = Move_State.move_piece(next_move, board)

    # determine white or black turn (1 for w, -1 for b)
    if game_state_i %2 == 1:
      x *= -1
    return x, y