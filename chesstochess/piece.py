import os
import chess
from clone_chess import Clone_Chess

class Piece:

    def __init__(self, name, color, value, img_uri=None, rectangle=None):
        self.clone_chess = Clone_Chess()
        self.name = name
        self.color = color

        value_sign = 1 if color == 'white' else -1
        self.value = value * value_sign
        self.ok_moves = []
        self.checklegal = []
        self.moved = False
        self.img_uri = img_uri
        self.set_texture()
        self.rectangle = rectangle

    def set_texture(self, size=80):
        self.img_uri = os.path.join(
            f'content/pieces/pieces_{size}px/{self.color}_{self.name}.png')

    def add_ok_move(self, move):
        self.ok_moves.append(move)
        if self.clone_chess.check_legal_move(move):

            self.checklegal.append(move)
        print(self.checklegal)

            
 


    def clear_moves(self):
        self.ok_moves = []

class Pawn(Piece):

    def __init__(self, color):
        self.dir = -1 if color == 'white' else 1
        self.en_passant = False
        self.pname = chess.WHITE if color == 'white' else chess.BLACK
        self.type = chess.PAWN
        super().__init__('pawn', color, 1.0)


class Knight(Piece):

    def __init__(self, color):
        self.pname = chess.WHITE if color == 'white' else chess.BLACK
        self.type = chess.KNIGHT
        super().__init__('knight', color, 3.0)


class Bishop(Piece):

    def __init__(self, color):
        self.pname = chess.WHITE if color == 'white' else chess.BLACK
        self.type = chess.BISHOP
        super().__init__('bishop', color, 3.001)


class Rook(Piece):

    def __init__(self, color):
        self.pname = chess.WHITE if color == 'white' else chess.BLACK
        self.type = chess.ROOK
        super().__init__('rook', color, 5.0)


class Queen(Piece):

    def __init__(self, color):
        self.pname = chess.WHITE if color == 'white' else chess.BLACK
        self.type = chess.QUEEN
        super().__init__('queen', color, 9.0)


class King(Piece):

    def __init__(self, color):
        self.pname = chess.WHITE if color == 'white' else chess.BLACK
        self.type = chess.KING
        self.left_rook = None
        self.right_rook = None
        super().__init__('king', color, 10000.0)