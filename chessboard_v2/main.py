import pygame
import sys

from base import *
from game import Game
from square import Square
from move import Move

class Getdata:

    def __init__(self):
        pygame.init()
        self.screenplay = pygame.display.set_mode((sp_width, sp_height))
        pygame.display.set_caption('Zama FHE Chess')
        self.game = Game()

    def snap(self):

        #screenplay = self.screenplay
        game = self.game
        #board = self.game.board
        #dragger = self.game.dragger
        """        i, j = 0,0
        while i < 8:
            while j < 8:"""
            # game.display_chessboard(screenplay)

            # # display last move
            # game.display_lastmove(screenplay)
            
            # # display move
            # game.display_moves(screenplay)

            # # display static pieces
            # game.display_pieces(screenplay)
        game.snapchot_pieces()
        


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
            # display chess board
            game.display_chessboard(screenplay)

            # display last move
            game.display_lastmove(screenplay)
            
            # display move
            game.display_moves(screenplay)

            # display static pieces
            game.display_pieces(screenplay)
            game.snapchot_pieces()
            # display user experience hover
            game.display_hover(screenplay)

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
                            board.compute_move(piece, selected_square_row, selected_square_col, bool=True)
                            dragger.save_source(event.pos)
                            dragger.drag_piece(piece)
                            game.display_chessboard(screenplay)
                            game.display_moves(screenplay)
                            game.display_pieces(screenplay)

                # mouse drags piece
                elif event.type == pygame.MOUSEMOTION:
                    game.set_hover(event.pos[1] // sqsize, event.pos[0] // sqsize)
                    
                    if dragger.dragging:
                        dragger.update_mouse(event.pos)
                        game.display_chessboard(screenplay)
                        game.display_lastmove(screenplay)
                        game.display_moves(screenplay)
                        game.display_pieces(screenplay)
                        game.display_hover(screenplay)
                        dragger.update_blit(screenplay)
                        
                
                # mouse releases piece
                elif event.type == pygame.MOUSEBUTTONUP:
                    
                    if dragger.dragging:
                        dragger.update_mouse(event.pos)

                        released_row = dragger.mouseY // sqsize
                        released_col = dragger.mouseX // sqsize

                        # get the squares for move
                        source = Square(dragger.source_row, dragger.source_col)
                        target = Square(released_row, released_col)

                        move = Move(source, target)

                        # check move ok ?
                        if board.valid_move(dragger.piece, move):
                            captured = board.squares[released_row][released_col].piece_presence()
                            board.move(dragger.piece, move)

                            board.set_true_en_passant(dragger.piece)
                            #game.sound_it(captured)
                            game.display_chessboard(screenplay)
                            game.display_lastmove(screenplay)
                            game.display_pieces(screenplay)
                            game.next_player()

                    dragger.undrag_piece()
                
                # reset app
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        game.reset()
                        game = self.game
                        board = self.game.board
                        dragger = self.game.dragger
                        
                # close app
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()




            pygame.display.update()


main = Main()
main.mainloop()

# board_data = Getdata()
# board_data.snap()