import chess
import chess.engine
from square import Square
from base import *
#-----python-chess-----#
"""
TO SPEED UP DEVELOPMENT AND FACILITATE AI DEVELOPMENT, USE OF PYTHON-CHESS
https://python-chess.readthedocs.io/en/latest/#
"""
class Clone_Chess:
    
    def __init__(self):
        self.board = chess.Board()

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

        self.piece_value = {"P":[100, self.pawn_ps], "N":[320, self.knight_ps], "B":[330, self.bishop_ps], "R":[500, self.rooks_ps], "Q":[900, self.queen_ps], "K":[10000, self.king_game_ps]}

    #       __ ___    __        __ 
    #  /\  /    |  | /  \ |\ | (_  
    # /--\ \__  |  | \__/ | \| __) 
                             
    
    # make a move from source to target square
    def move_clone_board(self, move):
       #uci_format = "".join((source,target))
       # checking the moves
       #if self.check_legal_move(move):
        uci_format = self.convert_move_2_string(move)
        self.board.push_san(uci_format)

    def move_clone_promotion(self, sq_s, sq_t, promotion):
        chess.Move(sq_s, sq_t, promotion)

    # clearing the board
    def clear_board(self):
        self.board.clear_board()
    
    def copy_board(self):
        return self.board.copy()
    
    def move_into_copy(self, move, copy_board):
        uci_format = self.convert_move_2_string(move)
        copy_board.push_san(uci_format)
    
    def clear_copy_board(self, copy_board):
        copy_board.clear_board()

    def piece_square_eval(self, copy_board):

        white_eval = 0
        for i in range(64):
            for k in self.piece_value.keys():
                if str(copy_board.piece_at(i)) == k:
                    white_eval += self.piece_value[k][0]+ self.piece_value[k][1][i]

        return white_eval

    #### GETS ####
    # get a snapshot of board
    def get_board(self):
        return self.board
    
    # get a FEN representation of the position
    def get_fen(self):
        return self.board.fen()

    # get the legal moves for a given position
    def legal_moves(self):
        return self.board.legal_moves

    # get the pseudo legal moves (might leaves or put the King in check)) for a given position
    def pseudo_legal_moves(self):
        return self.board.pseudo_legal_moves

    # get the number of legal moves
    def legal_moves_count(self):
        return self.board.legal_moves.count()

    # get game outcome
    def outcome(self, board):
        return board.outcome()


    def stockfish_evaluation(board, time_limit = 0.01):
        engine = chess.engine.SimpleEngine.popen_uci("/usr/local/Cellar/stockfish/16/bin/stockfish")
        result = engine.analyse(board, chess.engine.Limit(time=time_limit))
        return result['score']

    #  __       __  __             __   __ 
    # /   |__| |_  /   |_/ | |\ | / _  (_  
    # \__ |  | |__ \__ | \ | | \| \__) __)

    # check if move within legal moves
    def check_legal_move(self, move):
        uci_format = self.convert_move_2_string(move)
        return chess.Move.from_uci(uci_format) in self.legal_moves()

    def check_pseudo_legal_move(self, move):
        uci_format = self.convert_move_2_string(move)
        return chess.Move.from_uci(uci_format) in self.pseudo_legal_moves()

    # check checkmate?
    def checkmate_check(self):
        return self.board.is_checkmate()
    
    # check stalemate?
    def check_stalemate(self):
        return self.board.is_stalemate()
    
    # check enough material?
    def check_insuffisant_material(self):
        return self.board.is_insufficient_material()
    
    # check game_over?
    def check_gameover(self):
        return self.board.is_game_over()
    
    # check once fiveold repetition or 75 moves without pawn push or capture?
    def check_repetitions(self):
        return (self.board.is_fivefold_repetition(), self.board.is_seventyfive_moves())


    #  __                         __   __ 
    # /   |    /\  | |\/| | |\ | / _  (_  
    # \__ |__ /--\ | |  | | | \| \__) __) 

    def claim_repetitions(self):
        if self.board.can_claim_threefold_repetition():
            return self.board.can_claim_threefold_repetition()
        
        if self.board.can_claim_fifty_moves():
            return self.board.can_claim_fifty_moves()
    
    def claim_draw(self):
        return self.board.can_claim_draw()
    

    
    #       __      __   __  __   __ 
    # |__| |_  |   |__) |_  |__) (_  
    # |  | |__ |__ |    |__ | \  __)              

    # convert square[col][row] to string for uci format
    def convert_move_2_string(self, move):

        source_col = Square.get_algeb_not(move.source.col)
        source_row = str(8-move.source.row)
        target_col = Square.get_algeb_not(move.target.col)
        target_row = str(8-move.target.row)

        str_move = "".join((source_col,source_row,target_col,target_row))

        return str_move
    
    # for testing
    def convert_move_2_string_bis(self, source_col, source_row, target_col, target_row):

        src_col = Square.get_algeb_not(source_col)
        src_row = str(8-source_row)
        trgt_col = Square.get_algeb_not(target_col)
        trgt_row = str(8-target_row)

        # str_source = "".join((source_col,source_row))
        # str_target = "".join((target_col,target_row))

        str_move = "".join((src_col,src_row,trgt_col,trgt_row))
        print(str_move)
        return str_move #str_source, str_target

    # for testing
    def convert_string_2_move(self, str_move):
        
        source_target = [x for x in (chess.Move.uci(str_move))]
        source_col,source_row,target_col,target_row = source_target[0], source_target[1], source_target[2], source_target[3]

        source_col = Square.convert_algeb_not(source_col)
        target_col = Square.convert_algeb_not(target_col)

         # micro location
        source = Square(source_row, source_col) 
        target = Square(target_row, target_col)
        
        # move at micro
        return source, target