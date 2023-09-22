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
    
    
    def move_clone_board(self, move):
        """ Makes a push of move from source to target square"""
        uci_format = self.convert_move_2_string(move)
        try:
            self.board.push_san(uci_format)
        except chess.IllegalMoveError:
            self.check_termination(self.get_board())
            

    def move_clone_promotion(self, sq_s, sq_t, promotion):
        """ Push a pawn promotion of move from source to target square """
        chess.Move(sq_s, sq_t, promotion)

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



    def get_board(self):
        """get current board"""
        return self.board
    
    def get_fen(self):
        """get the FEN representation of current move"""
        return self.board.fen()

    def legal_moves(self):
        """get legal moves generator"""
        return self.board.legal_moves

    def pseudo_legal_moves(self):
        """get pseudo legal moves generator (might leave or put the King in check)"""
        return self.board.pseudo_legal_moves

    def legal_moves_count(self):
        """get the number of legal moves"""
        return self.board.legal_moves.count()

    def outcome(self, board):
        """get game outcome"""
        return board.outcome()


    # ⒸⒽⒺⒸⓀⓈ 🅒🅗🅔🅒🅚🅢 ⒸⒽⒺⒸⓀⓈ


    def check_legal_move(self, move):
        """checks if move within legal moves"""
        uci_format = self.convert_move_2_string(move)
        return chess.Move.from_uci(uci_format) in self.legal_moves()

    def check_pseudo_legal_move(self, move):
        """checks if move within pseudo legal moves"""
        uci_format = self.convert_move_2_string(move)
        return chess.Move.from_uci(uci_format) in self.pseudo_legal_moves()

    def checkmate_check(self):
        """checks checkmate?"""
        return self.board.is_checkmate()
    
    def check_stalemate(self):
        """checks stalemate?"""
        return self.board.is_stalemate()
    
    def check_insuffisant_material(self):
        """checks enough material?"""
        return self.board.is_insufficient_material()
    
    def check_gameover(self):
        """checks game_over?"""
        return self.board.is_game_over()
    
    def check_repetitions(self):
        """checks once fiveold repetition or 75 moves without pawn push or capture?"""
        return (self.board.is_fivefold_repetition(), self.board.is_seventyfive_moves())

    def check_termination(self, current_board):
        """checks all checks"""

        if current_board.outcome():
            if current_board.outcome().winner == chess.WHITE:
                print("White wins by %s" % current_board.outcome().termination)
            elif current_board.outcome().winner == chess.BLACK:
                print("Black wins %s" % current_board.outcome().termination)
            else:
                print("Draw, no winner nor looser.")
        print("Game %s" % current_board.outcome())



        """
        elif self.board.is_checkmate():
            print("is Check: %s" % current_board.is_checkmate())
            return True
        
        elif current_board.is_stalemate():
            print("is Stalemate: %s" % current_board.is_stalemate())
            return True

        elif current_board.is_insufficient_material():
            print("Insufficient_material: %s" % current_board.is_insufficient_material())
            return True
        
        elif current_board.is_game_over():
            print("Game_over: %s" % current_board.is_game_over())
            return True

        elif current_board.is_fivefold_repetition():
            print("Repetition 5: %s" % current_board.is_fivefold_repetition())
            return True
        
        elif current_board.is_seventyfive_moves():
            print("Repetition 75: %s" % current_board.is_seventyfive_moves())
            return True
        """


    #  ⒸⓁⒶⒾⓂⓈ 🅒🅛🅐🅘🅜🅢 ⒸⓁⒶⒾⓂⓈ

    def claim_repetitions(self):
        if self.board.can_claim_threefold_repetition():
            return self.board.can_claim_threefold_repetition()
        
        if self.board.can_claim_fifty_moves():
            return self.board.can_claim_fifty_moves()
    
    def claim_draw(self):
        return self.board.can_claim_draw()
    
    
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

    def convert_move_2_string_bis(self, source_col, source_row, target_col, target_row):
        """
        convert homemade (source[col][row] to target[col][row]) to string for uci format (used by Python-Chess library)
        """
        src_col = Square.get_algeb_not(source_col)
        src_row = str(8-source_row)
        trgt_col = Square.get_algeb_not(target_col)
        trgt_row = str(8-target_row)

        # str_source = "".join((source_col,source_row))
        # str_target = "".join((target_col,target_row))

        str_move = "".join((src_col,src_row,trgt_col,trgt_row))
        print(str_move)
        return str_move #str_source, str_target


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
    

    def stockfish_evaluation(board, time_limit = 0.01):
        engine = chess.engine.SimpleEngine.popen_uci("/usr/local/Cellar/stockfish/16/bin/stockfish")
        result = engine.analyse(board, chess.engine.Limit(time=time_limit))
        return result['score']