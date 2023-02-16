import re
import numpy as np

num_to_alpha = {0:"a", 1:"b", 2:"c",  3:"d",  4:"e",  5:"f",  6:"g",  7:"h"}
alpha_to_num = {"a":0, "b":1, "c":2,  "d":3,  "e":4,  "f":5,  "g":6,  "a":7}

""" codeflow
get moves from db
create empty matrix for each piece type
translate to matrix
"""

class Board_State():
    def __init__(self):
        pass

    def feat_map_piece(self, board, type):
        """convert board chess lib format to binary like"""
        self.sub_board = str(board)
        self.sub_board = re.sub(f'[^{type}{type.upper()} \n', '.', self.sub_board)
        self.sub_board = re.sub(f'[^{type}','-1', self.sub_board)
        self.sub_board = re.sub(f'[^{type.upper()}','1', self.sub_board)
        self.sub_board = re.sub(f'.','0', self.sub_board)

        board_matrix = []
        for row in self.sub_board.split('\n'):
            row = row.split(' ')
            row = [int(x) for x in row]
            board_matrix.append(row)

        return np.array(board_matrix) # numpy matrix


    def board_2_rep(self, board):
        """board to matrix representation per pieces types"""
        pieces = ['p','r','n','b','q','k']
        layers = []
        for piece in pieces:
            layers.append(self.feat_map_piece(board, piece)) # return feature map / pieces
        
        board_rep = np.stack(layers) #3D tensor
        return board_rep
    

class Move_State():
    """
    2 matrices for spatial features:
    #1 which piece to move from where
    #2 where to move the piece
    """

    def __init__(self):
        pass

    def move_piece(move, board):
        """function for moving"""
        board.push_san(move).uci() # 1st needs to convert the dataset from algebraic to uci format
        move = str(board.pop())

        initial_output_layer = np.zeros((8,8)) # from 1 to 0 on the departure matrix
        initial_row = 8 - int(move[1])
        initial_column = alpha_to_num[move[0]]
        initial_output_layer[initial_row,  initial_column] = 1

        destination_output_layer = np.zeros((8,8)) # from 1 to 0 on the arrival matrix
        destination_row = 8 - int(move[3])
        destination_column = alpha_to_num[move[2]]
        destination_output_layer[destination_row, destination_column] = 1

        return np.stack([initial_output_layer, destination_output_layer])
    

    def list_move_sequence(listms):
        """individual moves"""
        return re.sub('\d*\.','',listms).split(' ')[:-1]