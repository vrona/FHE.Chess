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
        board = self.Board()

        while True:
            game.show_backg(screenplay)
            game.show_pieces(screenplay)
            for event in pygame.event.get():

                # mouse selects piece
                if event.type == pygame.MOUSEBUTTONDOWN:
                    dragger.update_mouse(event.pos)
                    
                    selected_square_row = dragger.mouseY // sqsize
                    selected_square_col = dragger.mouseX // sqsize

                    # presence of piece within selected square
                    if board.squares[selected_square_row][selected_square_col].piece_presence():
                        piece = board.squares[selected_square_row][selected_square_col].piece

                # mouse drags piece
                elif event.type == pygame.MOUSEMOTION:
                    pass
                
                # mouse releases piece
                elif event.type == pygame.MOUSEBUTTONUP:
                    pass
                
                # close app
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()




            pygame.display.update()


main = Main()
main.mainloop()