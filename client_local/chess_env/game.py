import pygame
from base import *
from board import Board
from dragger import Dragger
from config import Config
from square import Square

class Game:

    def __init__(self):
        self.player_turn = 'white'
        self.square_hovered = None
        self.board = Board()
        self.dragger = Dragger()
        self.config = Config()

    def display_chessboard(self, surface):
        for row in range(cb_rows):
            for col in range(cb_cols):
                self.display_rect('#ffcb00', '#383E42', col, row, surface) 

                # numeric labels along col 0
                if col == 0:
                    numeric_label = self.config.font.render(str(cb_rows-row), 10, "#FFFFFF")
                    numeric_label_position = (5, 5+row * sqsize)
                    surface.blit(numeric_label, numeric_label_position)

                # numeric labels along col 0
                if row == 7:
                    algebraic_label = self.config.font.render(Square.get_algeb_not(col), 1,"#FFFFFF")
                    algebraic_label_position = (col * sqsize + sqsize - 20, sp_height - 20)
                    surface.blit(algebraic_label, algebraic_label_position)

    def display_pieces(self, surface):
        for row in range(cb_rows):
            for col in range(cb_cols):
                #presence of a piece
                if self.board.squares[row][col].piece_presence():

                    piece = self.board.squares[row][col].piece

                    if piece is not self.dragger.piece:
                        piece.set_texture(size=80)
                        # pygame specification
                        img = pygame.image.load(piece.img_uri)
                        img_center = col * sqsize + sqsize // 2, row * sqsize + sqsize // 2
                        piece.rectangle = img.get_rect(center=img_center)
                        surface.blit(img, piece.rectangle)


    def display_moves(self, surface):
        if self.dragger.dragging:
            piece = self.dragger.piece

            # ok moves show as image
            black_dots = pygame.image.load("client_local/content/imgdot/black_dots.png")
            white_dots = pygame.image.load("client_local/content/imgdot/white_dots.png")

            for move in piece.ok_moves:

                img_center = move.target.col * sqsize + sqsize // 2, move.target.row * sqsize + sqsize // 2

                if (move.target.row + move.target.col) % 2 == 0:

                    piece.rectangle = black_dots.get_rect(center=img_center)
                    surface.blit(black_dots, piece.rectangle)
                else:
                    piece.rectangle = white_dots.get_rect(center=img_center)
                    surface.blit(white_dots, piece.rectangle)
                
                # ok moves show as color self.display_rect('#ffeac8', '#ffebc6', move.target.col, move.target.row, surface, stroke=10)
    
    # this enable players to see where the move is from light grey
    def display_lastmove(self, surface):
        if self.board.last_move:
            source = self.board.last_move.source
            target = self.board.last_move.target

            for coor in [source, target]:
                self.display_rect('#f5f7f9', '#c1c4c8', coor.col, coor.row, surface)


    # when grabbing piece, it hovers over the chessboard
    def display_hover(self, surface):
        if self.square_hovered:
            self.display_rect('#ffeac8', '#ffebc6', self.square_hovered.col, self.square_hovered.row, surface, stroke=3)      

    # helper function for repetitive and necessary function in pygame
    def display_rect(self, ok_color, no_color, xcol, yrow, surface, stroke=0):
        
        color = ok_color if (yrow + xcol) % 2 == 0 else no_color # color
        rect = (xcol * sqsize, yrow * sqsize, sqsize, sqsize)   # rect
        pygame.draw.rect(surface, color, rect, stroke)         # blit

    def set_hover(self, row, col):
        self.square_hovered = self.board.squares[row][col]

    def next_player(self):
        self.player_turn = 'white' if self.player_turn == "black" else "black"

    def get_turn_color(self):
        print(self.player_turn)


    def reset(self):
        self.__init__()