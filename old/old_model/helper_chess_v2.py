import re
import chess
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
square_table = np.array([
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
    

    def feat_map_piece(self, board, color):
        """convert board chess lib format to binary like"""
        type, hotone = self.whattype(color)

        sub_board = str(board)
        sub_board = re.sub(f'[^{type} \n]', '.', sub_board)
        sub_board = re.sub(f'{type}', '{}'.format(hotone), sub_board)
        sub_board = re.sub(f'\.', '0', sub_board)
        board_matrix = []
        for row in sub_board.split('\n'):
            row = row.split(' ')
            row = [int(x) for x in row]
            board_matrix.append(row)

        return np.array(board_matrix) # numpy matrix


    def board_tensor(self, board):
        """board to matrix representation per pieces types and then stacked"""
        pieces = ['p','r','n','b','q','k','P', 'R', 'N', 'B', 'Q', 'K']
        layers = []
        for piece in pieces:
            layers.append(self.feat_map_piece(board, piece)) # return feature map / pieces
        
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

        square_location = square_table[choosen_piece_row, choosen_piece_column]
        piece = board.piece_at(square_location)
        #print(from_to_move,"\nCOLUMN {}:".format(from_to_move[0]),choosen_piece_column, "ROW {}:".format(from_to_move[1]),choosen_piece_row)
        #print("SQUARE:",square_location)
        #print("PIECE:",piece)
        return square_location, piece

    def choose_piece(self, move, board):
        """choosing the adequate piece to play"""
        board.push_san(move).uci() # 1st needs to convert the dataset from algebraic to uci format

        from_to_move = str(board.pop())

        #initial_output_layer = np.zeros((8,8)) # from 0 to 1 on the departure matrix
        initial_row = 8 - int(from_to_move[1])
        initial_column = alpha_to_num[from_to_move[0]]
        #initial_output_layer[initial_row,  initial_column] = 1
        
        square_location = square_table[initial_row, initial_column]
        piece = board.piece_at(square_location)

        # XOR binary square_location ^ piece_binary
        y=int(dict_sq_binary[square_location],2) ^ int(dict_piece_binary[str(piece)],2)
        y = '{0:b}'.format(y)

        bin_sq_loc_piece_array = np.zeros(shape=(8,))
        for i, z in enumerate(y):
            bin_sq_loc_piece_array[i] = int(z)

        # y_hat = torch.tensor(bin_sq_loc_piece_array, dtype=torch.int8)
        # print(y_hat)
        
        return bin_sq_loc_piece_array # square_table[initial_row, initial_column] #initial_output_layer


    def move_piece(self, move, board):
        """function for moving"""
        board.push_san(move).uci() # 1st needs to convert the dataset from algebraic to uci format

        move = str(board.pop())

        initial_output_layer = np.zeros((8,8)) # from 0 to 1 on the departure matrix
        initial_row = 8 - int(move[1])
        initial_column = alpha_to_num[move[0]]
        initial_output_layer[initial_row,  initial_column] = 1

        destination_output_layer = np.zeros((8,8)) # from 0 to 1 on the arrival matrix
        destination_row = 8 - int(move[3])
        destination_column = alpha_to_num[move[2]]
        destination_output_layer[destination_row, destination_column] = 1

        return np.stack([initial_output_layer, destination_output_layer])
    

    def list_move_sequence(self, listms):
        """individual moves"""
        return re.sub('\d*\. ','',listms).split(' ')[:-1]