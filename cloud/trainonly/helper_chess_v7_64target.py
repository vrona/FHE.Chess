import re
import numpy as np

# dictionary of algebraic(string) = digit
#num_to_alpha = {0:"a", 1:"b", 2:"c",  3:"d",  4:"e",  5:"f",  6:"g",  7:"h"}
alpha_to_num = {"a":0, "b":1, "c":2,  "d":3,  "e":4,  "f":5,  "g":6,  "h":7}


# array of square table location within chessboard (8x8) 
bitboard = np.array([
    [56,57,58,59,60,61,62,63],
    [48,49,50,51,52,53,54,55],
    [40,41,42,43,44,45,46,47],
    [32,33,34,35,36,37,38,39],
    [24,25,26,27,28,29,30,31],
    [16,17,18,19,20,21,22,23],
    [8,9,10,11,12,13,14,15],
    [0,1,2,3,4,5,6,7],
    ])


class Board_State():
    def __init__(self):
        pass
    
    def whattype(self, color):
        if color.isupper():
            value = 1
            return color.upper(), value
        else:
            value = -1
            return color, value
    

    def feat_map_piece_6(self, board, color):
            """convert board chess lib format to binary like"""

            sub_board = str(board)
            sub_board = re.sub(f'[^{color}{color.upper()} \n]', '.', sub_board)
            sub_board = re.sub(f'{color}', '{}'.format(-1), sub_board)
            sub_board = re.sub(f'{color.upper()}', '{}'.format(1), sub_board)
            sub_board = re.sub(f'\.', '0', sub_board)
            board_matrix = []
            for row in sub_board.split('\n'):
                row = row.split(' ')
                row = [int(x) for x in row]
                board_matrix.append(row)

            return np.array(board_matrix) # numpy matrix

    def feat_map_piece_12(self, board, color):
        """convert board chess lib format to binary like"""
        
        #type, hotone = self.whattype(color)
        t, v = self.whattype(color)
        
        sub_board = str(board)
        sub_board = re.sub(f'[^{t} \n]', '.', sub_board)
        sub_board = re.sub(f'{t}', '{}'.format(v), sub_board)
        sub_board = re.sub(f'\.', '0', sub_board)
        board_matrix = []
        for row in sub_board.split('\n'):
            row = row.split(' ')
            row = [int(x) for x in row]
            board_matrix.append(row)

        return np.array(board_matrix) # numpy matrix


    def board_tensor_6(self, board):
        """board to matrix representation per pieces types and then stacked"""
        pieces = ['p','r','n','b','q','k']
        layers = []
        for piece in pieces:
            layers.append(self.feat_map_piece_6(board, piece)) # return feature map / pieces

        board_rep = np.stack(layers) #3D tensor shape (6,8,8)
        return board_rep


    def board_tensor_12(self, board):
        """board to matrix representation per pieces types and then stacked"""
        pieces = ['p','r','n','b','q','k','P', 'R', 'N', 'B', 'Q', 'K']
        layers = []
        for piece in pieces:
            layers.append(self.feat_map_piece_12(board, piece)) # return feature map / pieces

        board_rep = np.stack(layers) #3D tensor shape (12,8,8)
        return board_rep

class Move_State():
    """
    #1 which square to move from
    #2 where square to move to
    """

    def __init__(self):
        pass
    

    def from_to_bitboards(self, move, board):
        """choosing the adequate square to target"""
        board.push_san(move).uci() # 1st needs to convert the dataset from algebraic to uci format

        from_to_move = str(board.pop())

        # source
        source_row = 8 - int(from_to_move[1])
        source_col = alpha_to_num[from_to_move[0]]

        source_square_bit = bitboard[source_row, source_col]
       
        source_flat_bit = np.zeros((64,))
        source_flat_bit[source_square_bit] = 1

        # target
        target_row = 8 - int(from_to_move[3])
        target_col = alpha_to_num[from_to_move[2]]

        target_square_bit = bitboard[target_row, target_col]
       
        target_flat_bit = np.zeros((64,))
        target_flat_bit[target_square_bit] = 1
        
        #print("SQUARE:",source_square_bit, target_square_bit)
        
        return source_flat_bit, target_flat_bit
    
    def source_flat_bit(self, source_square):
        """
        convert source square into flat (64,) array
        """
        source_flat_bit = np.zeros((64,))
        source_flat_bit[source_square] = 1
        return source_flat_bit


    def list_move_sequence(self, listms):
        """individual moves"""
        return re.sub('\d*\. ','',listms).split(' ')[:-1]
