import re
import numpy as np

# dictionary of algebraic(string) = digit
#num_to_alpha = {0:"a", 1:"b", 2:"c",  3:"d",  4:"e",  5:"f",  6:"g",  7:"h"}
alpha_to_num = {"a":0, "b":1, "c":2,  "d":3,  "e":4,  "f":5,  "g":6,  "h":7}

# dictionary of piece (string): binary(string) 8 bits
dict_piece_binary = {
"p":'01110000', "r":'01110010', "n":'01101110', "b":'01100010', "q":'01110001', "k":'01101011',
"P":'01010000', "R":'01010010', "N":'01001110', "B":'01000010', "Q":'01010001', "K":'01001011'
}

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

# dictionary of square_location(in): binary(string)
dict_sq_binary = {}
for z in range(0, 64):
    dict_sq_binary[z] = bin(z)[2:]

# dictionary of square_location(in): array from binary 8 bits (of square indices)
dict_array_sqbin = {}
for z in range(0, 64):
    dict_array_sqbin[z] = np.array(list(np.binary_repr(z).zfill(8))).astype(np.int8)

""" codeflow
get moves from db
create empty matrix for each piece type
translate to matrix
"""

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
    2 matrices for spatial features:
    #1 which piece to move from where
    #2 where to move the piece
    """

    def __init__(self):
        pass
    
    def piece_n_sqlocation(self, move, board):

        from_to_move = str(board.pop())
        choosen_piece_row = 8 - int(from_to_move[1]) # flipping the board (origin of row starts from top instead of bottom)
        choosen_piece_column = alpha_to_num[from_to_move[0]]

        square_location = bitboard[choosen_piece_row, choosen_piece_column]
        piece = board.piece_at(square_location)
        #print(from_to_move,"\nCOLUMN {}:".format(from_to_move[0]),choosen_piece_column, "ROW {}:".format(from_to_move[1]),choosen_piece_row)
        #print("SQUARE:",square_location)
        #print("PIECE:",piece)
        return square_location, piece


    def from_to_bitboards(self, move, board):
        """choosing the adequate piece to play"""
        board.push_san(move).uci() # 1st needs to convert the dataset from algebraic to uci format

        from_to_move = str(board.pop())

        # SQUARE_FROM
        from_row = 8 - int(from_to_move[1])
        from_column = alpha_to_num[from_to_move[0]]

        square_from = bitboard[from_row, from_column]
        vector_from = dict_array_sqbin[square_from]
       

        # SQUARE_TO
        # to_row = 8 - int(from_to_move[3])
        # to_column = alpha_to_num[from_to_move[2]]

        # square_to = bitboard[to_row, to_column]
        # vector_to = dict_array_sqbin[square_to]

        return vector_from
        
   

    def list_move_sequence(self, listms):
        """individual moves"""
        return re.sub('\d*\. ','',listms).split(' ')[:-1]