import pygame
import sys

from base import *
from game import Game
from dragger import Dragger
from board import Board
from square import Square
from move import *

class Main:

    def __init__(self):
        pygame.init()
        self.screenplay = pygame.display.set_mode((sp_width, sp_height))
        pygame.display.set_caption('Zama FHE Chess')
        self.game = Game()


    def mainloop(self):
        
        screenplay = self.screenplay
        game = self.game
        board = self.game.board
        dragger = self.game.dragger

        while True:
            # show chess board
            game.display_chessboard(screenplay)

            # show last move
            game.display_lastmove(screenplay)
            
            # show move
            game.display_moves(screenplay)

            # display static pieces
            game.display_pieces(screenplay)

            # display grabbed piece
            if dragger.dragging:
                dragger.update_blit(screenplay)


            for event in pygame.event.get():

                # mouse selects piece
                if event.type == pygame.MOUSEBUTTONDOWN:
                    dragger.update_mouse(event.pos)
                    
                    selected_square_row = dragger.mouseY // sqsize
                    selected_square_col = dragger.mouseX // sqsize

                    # presence of piece within selected square
                    if board.squares[selected_square_row][selected_square_col].piece_presence():
                        piece = board.squares[selected_square_row][selected_square_col].piece

                        if piece.color == game.player_turn:
                            board.compute_move(piece, selected_square_row, selected_square_col)
                            dragger.save_initial(event.pos)
                            dragger.drag_piece(piece)
                            game.display_chessboard(screenplay)
                            game.display_moves(screenplay)
                            game.display_pieces(screenplay)

                # mouse drags piece
                elif event.type == pygame.MOUSEMOTION:
                    if dragger.dragging:
                        dragger.update_mouse(event.pos)
                        game.display_chessboard(screenplay)
                        game.display_moves(screenplay)
                        game.display_pieces(screenplay)
                        dragger.update_blit(screenplay)
                        
                
                # mouse releases piece
                elif event.type == pygame.MOUSEBUTTONUP:
                    
                    if dragger.dragging:
                        dragger.update_mouse(event.pos)

                        released_row = dragger.mouseY // sqsize
                        released_col = dragger.mouseX // sqsize

                        # get the squares for move
                        initial = Square(dragger.initial_row, dragger.initial_col)
                        destination = Square(released_row, released_col)

                        move = Move(initial, destination)

                        # check move ok ?
                        if board.valid_move(dragger.piece, move):
                            board.move(dragger.piece, move)

                            game.display_chessboard(screenplay)
                            game.display_lastmove(screenplay)
                            game.display_pieces(screenplay)
                            game.next_player()

                    dragger.undrag_piece()
                
                # close app
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()




            pygame.display.update()


main = Main()
main.mainloop()