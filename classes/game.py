import pygame
from base import *
from board import Board
from dragger import Dragger
from config import Config


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
                self.display_rect('#ffcb00', '#000000', col, row, surface)

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
            black_dots = pygame.image.load("content/imgdot/black_dots.png")
            white_dots = pygame.image.load("content/imgdot/white_dots.png")

            for move in piece.ok_moves:
                img_center = move.destination.col * sqsize + sqsize // 2, move.destination.row * sqsize + sqsize // 2

                if (move.destination.row + move.destination.col) % 2 == 0:

                    piece.rectangle = black_dots.get_rect(center=img_center)
                    surface.blit(black_dots, piece.rectangle)
                else:
                    piece.rectangle = white_dots.get_rect(center=img_center)
                    surface.blit(white_dots, piece.rectangle)
                
                # ok moves show as color self.display_rect('#ffeac8', '#ffebc6', move.destination.col, move.destination.row, surface, stroke=10)
               
    def display_lastmove(self, surface):
        if self.board.last_move:
            initial = self.board.last_move.initial
            destination = self.board.last_move.destination

            for coor in [initial, destination]:
                self.display_rect('#f5f7f9', '#c1c4c8', coor.col, coor.row, surface)

    def display_hover(self, surface):
        if self.square_hovered:
            self.display_rect('#ffeac8', '#ffebc6', self.square_hovered.col, self.square_hovered.row, surface, stroke=3)      

    def display_rect(self, ok_color, no_color, xcol, yrow, surface, stroke=0):
        
        color = ok_color if (yrow + xcol) % 2 == 0 else no_color # color
        rect = (xcol * sqsize, yrow * sqsize, sqsize, sqsize)   # rect
        pygame.draw.rect(surface, color, rect, stroke)         # blit

    def set_hover(self, row, col):
        self.square_hovered = self.board.squares[row][col]

    def next_player(self):
        self.player_turn = 'white' if self.player_turn == "black" else "black"

    def sound_it(self, captured=False):
        if captured:
            self.config.capture_sound.play()
        else:
            self.config.move_sound.play()

    def reset(self):
        self.__init__()