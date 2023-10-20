import pygame
import sys
import chess

sys.path.append("client_local/")
from chess_network import Network

from base import sp_width, sp_height, sqsize
from game import Game
from square import Square
from move import Move
from clone_chess import Clone_Chess
from button import Button

class Main:

    def __init__(self):

        self.cs_network = Network()    
        pygame.init()
        self.screenplay = pygame.display.set_mode((sp_width, sp_height))
        pygame.display.set_caption('Zama FHE.Chess App.')

        self.game = Game()
        self.button = Button(self.cs_network.devmode)
        self.clone_chess = Clone_Chess()
        self.game_over = False
        self.game_count = 0        

    def outcome(self):
        screenplay = self.screenplay
        button = self.button
        clone_chess = self.clone_chess
        cs_network = self.cs_network 

        if clone_chess.get_board().outcome() is not None:
            button.white_ai = False
            button.black_ai = False
            
            while button.restart:
                button.button_restart(screenplay)                                  
                if clone_chess.get_board().outcome().winner == chess.WHITE:
                    button.show_result(screenplay, "White wins", "%s" % clone_chess.get_board().outcome().termination)
                    #print("White wins by %s" % clone_chess.get_board().outcome().termination)
                elif clone_chess.get_board().outcome().winner == chess.BLACK:
                    button.show_result(screenplay, "Black wins", "%s" % clone_chess.get_board().outcome().termination)
                    #print("Black wins %s" % clone_chess.get_board().outcome().termination)
                else:
                    button.show_result(screenplay, "Draw", "%s" % clone_chess.get_board().outcome().termination)
                    #print("Draw, no winner nor looser.")
                
                return True

    def AI_game_over(self, text):
        screenplay = self.screenplay
        button = self.button

        while button.restart:
            button.button_restart(screenplay)                                              
            button.show_AI_givingup(screenplay, text) #"AI made a wrong move."
        
            return True
    
    def ai_server(self, black=False):
        screenplay = self.screenplay
        game = self.game
        button = self.button
        board = self.game.board
        dragger = self.game.dragger
        clone_chess = self.clone_chess
        cs_network = self.cs_network
        ai_name = "Black" if black==True else "White"

        # get the snapshot of the board and use it as input_data to AI via server
        # get reply from server as list of tuples of moves

        chessboard = clone_chess.get_board(mirror=True) if black == True else clone_chess.get_board()

        listoftuplesofmoves = cs_network.send(chessboard)

        """
        Uses only the 1st tuple in listoftuplesofmoves as it supposed to be the best inferred move.
        """
        if listoftuplesofmoves is not None:
            selected_square_row = listoftuplesofmoves[0][0][1]
            selected_square_col = listoftuplesofmoves[0][0][0]
            targeted_square_row = listoftuplesofmoves[0][1][1]
            targeted_square_col = listoftuplesofmoves[0][1][0]

            # apply the move
            self.autonomous_piece(ai_name, 7-selected_square_row, selected_square_col, 7-targeted_square_row, targeted_square_col, board, game, clone_chess, dragger, button, screenplay, black)

        else:
            if self.AI_game_over("%s AI cannot infer any proposals." % ai_name ):
                self.game_over = True

    def mainloop(self):
        
        screenplay = self.screenplay
        game = self.game
        button = self.button
        board = self.game.board
        dragger = self.game.dragger
        clone_chess = self.clone_chess

        self.game_count += 1
        print("\n--Game %s has started--\n"%self.game_count)
        
        while True:

            # display chess board
            game.display_chessboard(screenplay)

            # display last move
            game.display_lastmove(screenplay)
            
            # display move
            game.display_moves(screenplay)

            # display static pieces
            game.display_pieces(screenplay)

            # display user experience hover
            game.display_hover(screenplay)

            # display buttons
            button.button_whiteAI(screenplay)
            button.button_blackAI(screenplay)
            button.button_AIvAI(screenplay)
            button.button_HvH(screenplay)

            # ⒶⒾ 🅐🅘 ⒶⒾ 🅐🅘 ⒶⒾ

            #AI vs AI 
            if button.is_white_ai_() and game.player_turn=="white": self.ai_server()
            if button.is_black_ai_() and game.player_turn=="black": self.ai_server(black=True)

            # get the outcome of game when not None
            if self.outcome() or self.game_over:
                if button.new_game:

                    game.reset()
                    game = self.game
                    board = self.game.board
                    dragger = self.game.dragger
                    clone_chess.reset_board()
                    button.normal = True
                    button.white_ai = False
                    button.black_ai = False
                    self.game_over = False
                    print("\n^^Game %s has been reseted^^\n"%self.game_count)
                    self.game_count += 1
                    print("\n--Game %s has started--\n"%self.game_count)

            # ⒽⓊⓂⒶⓃ 🅗🅤🅜🅐🅝 ⒽⓊⓂⒶⓃ
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

                        if piece.color == game.player_turn: #and button.is_color_human_(game.player_turn)

                            board.compute_move(piece, selected_square_row, selected_square_col)
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
                        #if board.valid_move(dragger.piece, move):
                        board.piece_legal(clone_chess.get_board(), dragger.piece)
                        if board.new_valid_move(dragger.piece, move):
                            board.move(dragger.piece, move)

                            # pawn promotion to queen
                            if dragger.piece.type == chess.PAWN and game.board.squares[released_row][released_col].piece.type == chess.QUEEN:
                                # BRIDGE HERE cloning move from app to python-chess
                                clone_chess.move_clone_board(move, to_promote=True)
                            else:
                                # BRIDGE HERE cloning move from app to python-chess
                                clone_chess.move_clone_board(move)
                            print("\n%s %s %s%s to %s%s" % (piece.color,piece.name,Square.algebraic_notation_cols[dragger.source_col], 7-dragger.source_row,Square.algebraic_notation_cols[released_col], 7-released_row))
                            # print(clone_chess.get_board())
                            
                            # uncomment to get FEN output
                            print("\n%s (Human) FEN: "% game.player_turn, clone_chess.get_fen())
                            print("---------------")
                            
                            board.set_true_en_passant(dragger.piece)
                            

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
                        clone_chess.reset_board()
                        button.normal = True
                        button.white_ai = False
                        button.black_ai = False
                        self.game_over = False
                        print("\n^^Game %s has been reseted^^\n"%self.game_count)
                        self.game_count += 1
                        print("\n--Game %s has started--\n"%self.game_count)
                        
                # close app
                elif event.type == pygame.QUIT:
                    print("\n**You have asked to quit**\n")
                    pygame.quit()
                    
                    sys.exit()

            pygame.display.update()
            

    def autonomous_piece(self,ai_name, source_row, source_col, target_row, target_col, board, game, clone_chess, dragger, button,surface,black):
        """Makes the AI's move inference applied into homemade chessboard environment"""
        
        # mirror chess grid for black point of view
        if black == True:
            source_row, target_row = 7-source_row, 7-target_row

        # presence of piece within selected square
        if self.game.board.squares[source_row][source_col].piece_presence():
            piece = self.game.board.squares[source_row][source_col].piece

            if piece.color == self.game.player_turn:
                
                board.compute_move(piece, source_row, source_col)

                # get the squares for move
                source = Square(source_row, source_col)
                target = Square(target_row, target_col)

                move = Move(source, target)

                #  check move ok ?
                board.piece_legal(clone_chess.get_board(),piece)

                if game.board.new_valid_move(piece, move):
                    board.move(piece, move)

                    if piece.type == chess.PAWN and game.board.squares[target_row][target_col].piece.type == chess.QUEEN:

                            # BRIDGE HERE cloning move from app to python-chess
                        clone_chess.move_clone_board(move, to_promote=True)
                    else:
                            # BRIDGE HERE cloning move from app to python-chess"""
                        clone_chess.move_clone_board(move)

                    board.set_true_en_passant(piece)

                    print("\n%s %s %s%s to %s%s" % (piece.color,piece.name,Square.get_algeb_not(source_col), 8-source_row,Square.get_algeb_not(target_col), 8-target_row))
                        
                    # uncomment to get FEN output
                    print("\n%s (AI) FEN: "% ai_name,clone_chess.get_fen())

                    game.display_chessboard(surface)
                    game.display_lastmove(surface)
                    game.display_pieces(surface)
                    game.next_player()
                
                else:
                    if self.AI_game_over("%s AI wrongly inferred: %s%s %s%s" % (ai_name,Square.algebraic_notation_cols[source_col], 8-source_row, Square.algebraic_notation_cols[target_col], 8-target_row)):
                        self.game_over = True
                                  
        else:
            if self.AI_game_over("Dear %s AI, you've missed it: no piece here %s%s %s%s" % (ai_name,Square.algebraic_notation_cols[source_col], 8-source_row, Square.algebraic_notation_cols[target_col], 8-target_row)):
                self.game_over = True

main = Main()
main.mainloop()