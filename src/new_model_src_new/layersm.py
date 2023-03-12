import chess
import numpy as np
import re
from helper_chess_new import Move_State

helper_move_state = Move_State()

alpha_to_num = {"a":0, "b":1, "c":2,  "d":3,  "e":4,  "f":5,  "g":6,  "h":7}
chessloc = np.array([
    [56,57,58,59,60,61,62,63],
    [48,49,50,51,52,53,54,55],
    [40,41,42,43,44,45,46,47],
    [32,33,34,35,36,37,38,39],
    [24,25,26,27,28,29,30,31],
    [16,17,18,19,20,21,22,23],
    [8,9,10,11,12,13,14,15],
    [0,1,2,3,4,5,6,7],
                     ])

game = "1. e4 e5 2. Nc3 c6 3. Nf3 Qe7 4. d4 exd4 5. Nxd4 f6 6. Bc4 g6 7. O-O a6 8. a4 b6 9. Re1 Kd8 10. e5 f5 11. e6 d5 12. Ba2 Bb7 13. Bf4 c5 14. Nf3 d4 15. Bg5 Nf6 16. Ne5 Qg7 17. Nf7+ Ke7 18. Nxh8 Qxh8 19. Bxf6+ Qxf6 20. Ne2 Qg5 21. f3 Nc6 22. Bd5 Qe3+ 23. Kh1 Qg5 24. Ng3 Rd8 25. c4 dxc3 26. bxc3 f4 27. Ne4 Qxd5 28. Qxd5 Rxd5 29. Rad1 Rxd1 30. Rxd1 Kxe6 31. Ng5+ Kf5 32. Nxh7 Bh6 33. Rd7 Ba8 34. Rc7 g5 35. h3 Ke6 36. Kg1 Kd6 37. Rf7 Kd5 38. Rf6 Bg7 39. Rg6 Bxc3 40. Rxg5+ Kd4 41. Nf6 b5 42. axb5 axb5 43. Ne4 Ne5 44. Nxc3 Kxc3 45. Rxe5 c4 46. Rxb5 Kd4 47. Rb8 Bc6 48. Rc8 Kd5 49. Rb8 c3 50. Rb1 Kd4 51. Rc1 Kd3 52. Kf2 Kd2 53. Rg1 c2 54. g3 fxg3+ 55. Kxg3 Bb5 56. h4 Be2 57. Ra1 c1=Q 58. h5 Qc7+ 59. f4 Qg7+ 60. Kh4 Qxa1 61. h6 Qf6+ 0-1"

sequence_moves = helper_move_state.list_move_sequence(game)
"""Prendre que les paires pour avoir que les coups des blancs, puis que les impairs pour avoir le coup des noires
puis via UCI la piece avant le move pour apprendre à choisir la pièce
puis prendre le move de destination"""

piece = sequence_moves[5]
moves = sequence_moves[:1]

board = chess.Board()


for move in moves:
    board.push_san(move).uci()

from_to_move = str(board.pop())

row = 8 - int(from_to_move[1]) # flipping the board (origin of row starts from top instead of bottom)
column = alpha_to_num[from_to_move[0]]

print(from_to_move,"\nCOLUMN {}:".format(from_to_move[0]),column, "ROW {}:".format(from_to_move[1]),row)
bitmap = chessloc[row, column]
print("SQUARE:",bitmap)
piece = board.piece_at(bitmap)
print("PIECE:",piece) #,board


initial_output_layer = np.zeros((8,8)) # from 0 to 1 on the departure matrix
initial_output_layer[row,  column] = 1
print(initial_output_layer)


############## DONE ################
# def whattype(color):
#     if color.isupper():
#         value = 1
#         return color.upper(), value

#     else:
#         value = -1
#         return color, value


# def feat_map_piece(board, color):
#     """convert board chess lib format to binary like"""
#     t, v = whattype(color)

#     sub_board = str(board)
#     sub_board = re.sub(f'[^{t} \n]', '.', sub_board)
    
#     sub_board = re.sub(f'{t}', '{}'.format(v), sub_board)

#     sub_board = re.sub(f'\.', '0', sub_board)

#     board_matrix = []
#     for row in sub_board.split('\n'):
#         row = row.split(' ')
#         row = [int(x) for x in row]
#         board_matrix.append(row)

#     return np.array(board_matrix) # numpy matrix

# def board_tensor(board):
#     """board to matrix representation per pieces types and then stacked"""
#     piece = ['p','r','n','b','q','k','P', 'R', 'N', 'B', 'Q', 'K']
#     layers = []
#     for piece in piece:
#         layers.append(feat_map_piece(board, piece)) # return feature map / pieces


#     board_rep = np.stack(layers) #3D tensor shape (12,8,8)
    
#     return board_rep

# boardgame = board_tensor(board) 

# print(boardgame, boardgame.shape)

