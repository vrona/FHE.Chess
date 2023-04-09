import os
import chess
from clone_chess import Clone_Chess

class Piece:

    def __init__(self, name, color, img_uri=None, rectangle=None):
        self.clone_chess = Clone_Chess()
        self.name = name
        self.color = color
        self.ok_moves = []
        self.moved = False
        self.img_uri = img_uri
        self.set_texture()
        self.rectangle = rectangle


    def set_texture(self, size=80):
        self.img_uri = os.path.join(
            f'client/content/pieces/pieces_{size}px/{self.color}_{self.name}.png')


    def add_ok_move(self, move):
        self.ok_moves.append(move)


    def clear_moves(self):
        self.ok_moves = []


class Pawn(Piece):

    def __init__(self, color):
        self.dir = -1 if color == 'white' else 1
        self.en_passant = False
        self.pname = chess.WHITE if color == 'white' else chess.BLACK
        self.type = chess.PAWN
        super().__init__('pawn', color)


class Knight(Piece):

    def __init__(self, color):
        self.pname = chess.WHITE if color == 'white' else chess.BLACK
        self.type = chess.KNIGHT
        super().__init__('knight', color)


class Bishop(Piece):

    def __init__(self, color):
        self.pname = chess.WHITE if color == 'white' else chess.BLACK
        self.type = chess.BISHOP
        super().__init__('bishop', color)


class Rook(Piece):

    def __init__(self, color):
        self.pname = chess.WHITE if color == 'white' else chess.BLACK
        self.type = chess.ROOK
        super().__init__('rook', color)


class Queen(Piece):

    def __init__(self, color):
        self.pname = chess.WHITE if color == 'white' else chess.BLACK
        self.type = chess.QUEEN
        super().__init__('queen', color)


class King(Piece):

    def __init__(self, color):
        self.pname = chess.WHITE if color == 'white' else chess.BLACK
        self.type = chess.KING
        self.left_rook = None
        self.right_rook = None
        super().__init__('king', color)