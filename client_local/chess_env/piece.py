import os
import chess

class Piece:

    def __init__(self, name, color, img_uri=None, rectangle=None):
        self.name = name
        self.color = color
        self.ok_moves = []
        self.legal_move = []
        self.dictofmove = {}
        self.moved = False
        self.img_uri = img_uri
        self.set_texture()
        self.rectangle = rectangle

    def set_texture(self, size=80):
        self.img_uri = os.path.join(
            f'client_local/content/pieces/pieces_{size}px/{self.color}_{self.name}.png')

    def add_ok_move(self, move):
        """adds to each piece its ok_moves based on its behaviour"""
        self.ok_moves.append(move)

    def pop_ok_move(self, move):
        """pop to each piece an ok_moves based on exceptional moves"""
        self.ok_moves.pop(self.ok_moves.index(move))

    def add_legalmove(self, move):
        """adds to each piece its ok_moves based on its behaviour"""
        self.legal_move.append(move)

    def dict_legal(self, source_sq):
        self.dictofmove[source_sq] = self.legal_move

    def clear_moves(self):
        self.ok_moves = []

    def clear_legal_move(self):
        self.legal_move = []

    def show_legal_move(self):
        return self.legal_move


class Pawn(Piece):

    def __init__(self, color):
        self.dir = -1 if color == 'white' else 1
        self.en_passant = False
        self.promoted = False
        self.pname = "P" if color == 'white' else "p"
        self.type = chess.PAWN
        super().__init__('pawn', color)


class Knight(Piece):

    def __init__(self, color):
        self.pname = "N" if color == 'white' else "n"
        self.type = chess.KNIGHT
        super().__init__('knight', color)


class Bishop(Piece):

    def __init__(self, color):
        self.pname = "B" if color == 'white' else "b"
        self.type = chess.BISHOP
        super().__init__('bishop', color)


class Rook(Piece):

    def __init__(self, color):
        self.pname = "R" if color == 'white' else "r"
        self.type = chess.ROOK
        super().__init__('rook', color)


class Queen(Piece):

    def __init__(self, color):
        self.pname = "Q" if color == 'white' else "q"
        self.type = chess.QUEEN
        self.is_promotion = False
        super().__init__('queen', color)


class King(Piece):

    def __init__(self, color):
        self.pname = "K" if color == 'white' else "k"
        self.type = chess.KING
        self.left_rook = None
        self.right_rook = None
        super().__init__('king', color)