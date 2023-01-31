import pygame
import sys

from base import *
from game import Game
from dragger import Dragger
from board import Board


class Main:

    def __init__(self):
        pygame.init()
        self.screenplay = pygame.display.set_mode((sp_width, sp_height))
        pygame.display.set_caption('Zama FHE Chess')
        self.game = Game()
        self.dragger = Dragger()
        self.board = Board()


    def mainloop(self):
        
        screenplay = self.screenplay
        game = self.game
        dragger = self.dragger
        board = self.board

        while True:
            # show chess board
            game.display_chessboard(screenplay)

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
                        dragger.save_initial(event.pos)
                        dragger.drag_piece(piece)
                        # game.display_chessboard(screenplay)
                        # game.display_pieces(screenplay)

                # mouse drags piece
                elif event.type == pygame.MOUSEMOTION:
                    if dragger.dragging:
                        dragger.update_mouse(event.pos)
                        game.display_chessboard(screenplay)
                        dragger.update_blit(screenplay)
                        game.display_pieces(screenplay)
                        
                
                # mouse releases piece
                elif event.type == pygame.MOUSEBUTTONUP:
                    dragger.undrag_piece()
                
                # close app
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()




            pygame.display.update()


main = Main()
main.mainloop()