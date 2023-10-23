import chess
import chess.engine
from square import Square
from base import *
#-----python-chess-----#
"""
TO SPEED UP DEVELOPMENT AND FACILITATE AI DEVELOPMENT, USE OF PYTHON-CHESS Library.
https://python-chess.readthedocs.io/en/latest/#

This script is called Clone_Chess as it clones what happened in the homemade chessboard into the Python-Chess chess.Board().
This script aimed to call module from and push data into the Python-Chess.
Recall that AI is nutured with data which are transformed chess.Board() current state.
"""
class Clone_Chess:
    
    def __init__(self):

        self.board = chess.Board()
        self.board_mirror = chess.Board().mirror()
        self.engine = chess.engine.SimpleEngine.popen_uci("/usr/local/Cellar/stockfish/16/bin/stockfish")

        #piece square tables and (material) value are from Chess Programming Wiki https://www.chessprogramming.org/Simplified_Evaluation_Function
        
        # piece square table (square value)
        self.pawn_ps = np.array([
            0,  0,  0,  0,  0,  0,  0,  0,5, 10, 10,-20,-20, 10, 10,  5,5, -5,-10,  0,  0,-10, -5,  5,0,  0,  0, 20, 20,  0,  0,  0,5,  5, 10, 25, 25, 10,  5,  5,10, 10, 20, 30, 30, 20, 10, 10,50, 50, 50, 50, 50, 50, 50, 50,0,  0,  0,  0,  0,  0,  0,  0
            ])

        self.knight_ps = np.array([
            -50,-40,-30,-30,-30,-30,-40,-50,-40,-20,  0,  5,  5,  0,-20,-40,-30,  5, 10, 15, 15, 10,  5,-30,-30,  0, 15, 20, 20, 15,  0,-30,-30,  5, 15, 20, 20, 15,  5,-30,-30,  0, 10, 15, 15, 10,  0,-30,-40,-20,  0,  0,  0,  0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50,
            ])

        self.bishop_ps = np.array([
            -20,-10,-10,-10,-10,-10,-10,-20,-10,  5,  0,  0,  0,  0,  5,-10,-10, 10, 10, 10, 10, 10, 10,-10,-10,  0, 10, 10, 10, 10,  0,-10,-10,  5,  5, 10, 10,  5,  5,-10,-10,  0,  5, 10, 10,  5,  0,-10,-10,  0,  0,  0,  0,  0,  0,-10,-20,-10,-10,-10,-10,-10,-10,-20,
            ])

        self.rooks_ps = np.array([
            0,  0,  0,  5,  5,  0,  0,  0, -5,  0,  0,  0,  0,  0,  0, -5, -5,  0,  0,  0,  0,  0,  0, -5, -5,  0,  0,  0,  0,  0,  0, -5, -5,  0,  0,  0,  0,  0,  0, -5, -5,  0,  0,  0,  0,  0,  0, -5, 5, 10, 10, 10, 10, 10, 10,  5, 0,  0,  0,  0,  0,  0,  0,  0,
            ])

        self.queen_ps = np.array([
            -20,-10,-10, -5, -5,-10,-10,-20,-10,  0,  5,  0,  0,  0,  0,-10,-10,  5,  5,  5,  5,  5,  0,-10,0,  0,  5,  5,  5,  5,  0, -5,-5,  0,  5,  5,  5,  5,  0, -5,-10,  0,  5,  5,  5,  5,  0,-10,-10,  0,  0,  0,  0,  0,  0,-10,-20,-10,-10, -5, -5,-10,-10,-20,
            ])

        self.king_game_ps = np.array([
            20, 30, 10,  0,  0, 10, 30, 20,20, 20,  0,  0,  0,  0, 20, 20, -10,-20,-20,-20,-20,-20,-20,-10,-20,-30,-30,-40,-40,-30,-30,-20,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30
            ])

        self.king_end_ps = np.array([
            -50,-30,-30,-30,-30,-30,-30,-50,-30,-30,  0,  0,  0,  0,-30,-30,-30,-10, 20, 30, 30, 20,-10,-30,-30,-10, 30, 40, 40, 30,-10,-30,-30,-10, 30, 40, 40, 30,-10,-30,-30,-10, 20, 30, 30, 20,-10,-30,-30,-20,-10,  0,  0,-10,-20,-30,-50,-40,-30,-20,-20,-30,-40,-50,
            ])

        # piece (material) value
        self.piece_value = {"P":[100, self.pawn_ps], "N":[320, self.knight_ps], "B":[330, self.bishop_ps], "R":[500, self.rooks_ps], "Q":[900, self.queen_ps], "K":[10000, self.king_game_ps]}



    # ⒶⒸⓉⒾⓄⓃⓈ 🅐🅒🅣🅘🅞🅝🅢 ⒶⒸⓉⒾⓄⓃⓈ
    
    
    def move_clone_board(self, move, mirror=False, to_promote=False):
        """ Makes a push of move from source to target square"""
        uci_format = self.convert_move_2_string(move)
        
        if to_promote:
           """ Push a pawn promotion of move from source to target square """
           uci_format = uci_format+"q"

        self.board_mirror.push_san(uci_format) if mirror==True else self.board.push_san(uci_format) 

    def reset_board(self):
        """ reset current board"""
        return self.board.reset()

    def clear_board(self):
        """clear current board"""
        self.board.clear_board()
    
    def copy_board(self):
        """copy current board"""
        return self.board.copy()
    
    def move_into_copy(self, move, copy_board):
        """push move into copy board"""
        uci_format = self.convert_move_2_string(move)
        copy_board.push_san(uci_format)
    
    def clear_copy_board(self, copy_board):
        """clear copy board"""
        copy_board.clear_board()

    def piece_square_eval(self, copy_board):
        """compute the values of white based on white pieces type and"""
        white_eval = 0
        for i in range(64):
            for k in self.piece_value.keys():
                if str(copy_board.piece_at(i)) == k:
                    white_eval += self.piece_value[k][0]+ self.piece_value[k][1][i]
        return white_eval


    # ⒼⒺⓉⓈ 🅖🅔🅣🅢 ⒼⒺⓉⓈ

    def get_board(self, mirror=False):
        """get current board"""
        return self.board.mirror()if mirror==True else self.board
        
    def get_fen(self):
        """get the FEN representation of current move"""
        return self.board.fen()

    def stockfish_evaluation(self,board):
        
        result = self.engine.analyse(board, chess.engine.Limit(depth=3))
        print(result)
        #return result['score']

    # ⒸⒽⒺⒸⓀⓈ 🅒🅗🅔🅒🅚🅢 ⒸⒽⒺⒸⓀⓈ


    # ⒸⓁⒶⒾⓂⓈ 🅒🅛🅐🅘🅜🅢 ⒸⓁⒶⒾⓂⓈ
    
    
    # ⒽⒺⓁⓅⒺⓇⓈ 🅗🅔🅛🅟🅔🅡🅢 ⒽⒺⓁⓅⒺⓇⓈ

    def convert_move_2_string(self, move):
        """
        convert homemade "move" to string for uci format (used by Python-Chess library)
        Recall that in homemade, move is based on from square[col][row] to square[col][row]
        """
        source_col = Square.get_algeb_not(move.source.col)
        source_row = str(8-move.source.row)
        target_col = Square.get_algeb_not(move.target.col)
        target_row = str(8-move.target.row)
        str_move = "".join((source_col,source_row,target_col,target_row))

        return str_move
    

    # ⓉⒺⓈⓉⓈ 🅣🅔🅢🅣🅢 ⓉⒺⓈⓉⓈ

    def convert_move_2_string_bis(self, source_col, source_row, target_col, target_row, to_promote=False):
        """
        convert homemade (source[col][row] to target[col][row]) to string for uci format (used by Python-Chess library)
        """
        src_col = Square.get_algeb_not(source_col)
        src_row = str(8-source_row)
        trgt_col = Square.get_algeb_not(target_col)
        trgt_row = str(8-target_row)

        # str_source = "".join((source_col,source_row))
        # str_target = "".join((target_col,target_row))
        if to_promote:
            str_move = "".join((src_col,src_row,trgt_col,trgt_row,"q"))
            print(str_move)
        else:
            str_move = "".join((src_col,src_row,trgt_col,trgt_row))
            print(str_move)
        #return str_move #str_source, str_target


    def convert_string_2_move(self, str_move):
        """opposite of """
        source_target = [x for x in (chess.Move.uci(str_move))]
        source_col,source_row,target_col,target_row = source_target[0], source_target[1], source_target[2], source_target[3]

        source_col = Square.convert_algeb_not(source_col)
        target_col = Square.convert_algeb_not(target_col)

         # micro location
        source = Square(source_row, source_col) 
        target = Square(target_row, target_col)
        
        # move at micro
        return source, target
