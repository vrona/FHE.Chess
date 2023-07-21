import pygame
import sys
import copy
import chess

sys.path.insert(1,"/Volumes/vrona_SSD/FHE.Chess/client/")
from chess_network import Network

#from en_de_crypt import EnDe_crypt
# sys.path.insert(1,"client/chess_env")
from base import sp_width, sp_height, sqsize, bitboard
from game import Game
from square import Square
from move import Move
from clone_chess import Clone_Chess
from button import Button

# sys.path.insert(1,"server/model")
# from inference_64bit import Inference


class Main:

    def __init__(self):
        pygame.init()
        self.screenplay = pygame.display.set_mode((sp_width, sp_height))
        pygame.display.set_caption('Zama FHE.Chess App.')
        self.game = Game()
        self.button = Button()
        self.clone_chess = Clone_Chess()
        #self.inference = Inference() in case to debug inference
        self.cs_network = Network()
        #self.ende_crypt = EnDe_crypt()
    

    def pawn_promotion(self, source_row, source_col, target_row, target_col):
            """ this is specific to chess-python library: let pawn promoted"""
            self.clone_chess.move_clone_promotion(bitboard[source_row, source_col], bitboard[target_row, target_col], chess.QUEEN)
    

    def mainloop(self):
        
        screenplay = self.screenplay
        game = self.game
        button = self.button
        board = self.game.board
        dragger = self.game.dragger
        clone_chess = self.clone_chess
        cs_network = self.cs_network


        #cs_network.send(clone_chess.get_board())
        #ende_crypt = self.ende_crypt


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


            button.button_whiteAI(screenplay)
            # button.button_blackAI(screenplay)
            # button.button_bothAI(screenplay)
            button.button_HH(screenplay)

            # AI PART
            if button.get_ai_mode() and game.player_turn=="white":

                # print the Outcome of the game
                #if board.check_termination(clone_chess.get_board()): break

                # get the snapshot of the board and use it as input_data to AI via server
                # get reply from server as list of tuples of moves
                chessboard = clone_chess.get_board()
                """ fen = clone_chess.get_fen()
                myboard = chess.Board(str(fen))

                engine = chess.engine.SimpleEngine.popen_uci("/usr/local/Cellar/stockfish/16/bin/stockfish")
                result = engine.analyse(myboard, chess.engine.Limit(depth=2), multipv=3)
                #print(result['score'])
                #best_move = result.BestMove()
                #print(result)"""
                listoftuplesofmoves = cs_network.send(chessboard)


                # if result['pv'][0] in listoftuplesofmoves:
                #     print("")
                
                """###### TO DELETE
                for i in listoftuplesofmoves:
                    #clone_chess.convert_move_2_string_bis(i[0][1])
                    print("Choix Multiple:",i)#,"-->",clone_chess.convert_move_2_string_bis(i[0][0], 7-i[0][1], i[1][0], 7 -i[1][1]))
                """


                # get_chessboard = EnDe_crypt(chessboard)
                # get_chessboard.predict()

                selected_square_row = listoftuplesofmoves[0][0][1]
                selected_square_col = listoftuplesofmoves[0][0][0]
                targeted_square_row = listoftuplesofmoves[0][1][1]
                targeted_square_col = listoftuplesofmoves[0][1][0]

                #source_r, source_c, target_r, target_c = self.autonomous_check_sim(listoftuplesofmoves)

                # making the move
                #self.autonomous_piece(source_r, source_c, target_r, target_c, board, game, clone_chess, screenplay)
                self.autonomous_piece(7-selected_square_row, selected_square_col, 7-targeted_square_row, targeted_square_col, board, game, clone_chess, screenplay)
                
            # HUMAN PART
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
                            
                            # print the Outcome of the game
                            #if board.check_termination(clone_chess.get_board()):
                            #    button.button_reset(screenplay)

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

                            board.move(dragger.piece, move)
                            
                            # pawn promotion to queen
                            if dragger.piece.type == chess.PAWN and game.board.squares[released_row][released_col].piece.type == chess.QUEEN:
                                
                                # BRIDGE HERE cloning move from app to python-chess
                                self.pawn_promotion(dragger.source_row, dragger.source_col, released_row, released_col)

                            else:
                                # BRIDGE HERE cloning move from app to python-chess
                                clone_chess.move_clone_board(move)
                                print("SEE HUMAN Record:\n",clone_chess.get_board())
                            
                            board.set_true_en_passant(dragger.piece)
                            
                            # uncomment to get FEN output
                            #print("main.py",clone_chess.convert_move_2_string(move))
                            #print("HUMAN FEN output: ",clone_chess.get_fen())
                            game.display_chessboard(screenplay)
                            game.display_lastmove(screenplay)
                            game.display_pieces(screenplay)
                                # print the Outcome of the game
                            print("Human ZONE --")
                            if board.check_termination(clone_chess.get_board()): break

                            game.next_player()


                    dragger.undrag_piece()
                    
             
                    
                    #print("Game outcome from Human", self.clone_chess.get_board().outcome())
                
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

    def autonomous_check_sim(self, listofmove):
            
        move_eval = {}

        """ if listofmove == 0:
            pass
            else:
        """

        for i in range(len(listofmove)):
            """"for simulation"""

            tempboard = copy.deepcopy(self.game.board)
            #tempboard.move(temppiece, move, simulation=True) # move virtually one piece
            temp_cloneboard = self.clone_chess.copy_board()
            
            self.clone_chess.piece_square_eval(temp_cloneboard)

            move = listofmove[i]

            source_row = 7 - move[0][1]
            source_col = move[0][0]
            target_row = 7 - move[1][1]
            target_col = move[1][0]

            if tempboard.squares[source_row][source_col].piece_presence():
                piece = tempboard.squares[source_row][source_col].piece

                if piece.color == self.game.player_turn:    
                    tempboard.compute_move(piece, source_row, source_col, bool=False)

                    # get the squares for move
                    source = Square(source_row, source_col)
                    target = Square(target_row, target_col)

                    move = Move(source, target)

                    #  check move ok ?
                    if not tempboard.valid_move(piece, move):
                        listofmove.pop(listofmove.index(listofmove[i]))
                        print("%s poped out" % self.clone_chess.convert_move_2_string(move))
                    
                    else:
                        self.clone_chess.move_into_copy(move,temp_cloneboard)
                        print("find move", self.clone_chess.convert_move_2_string(move))

                        move_eval[source_row, source_col, target_row, target_col] = self.clone_chess.piece_square_eval(temp_cloneboard)
                        #self.clone_chess.clear_copy_board(temp_cloneboard)


        vlist = []
        klist = []
        while len(move_eval) > 0:
            vlist.append(max(move_eval.values()))
            klist.append(max(move_eval,key=move_eval.get))
            move_eval.pop(max(move_eval,key=move_eval.get))

        return klist[0][0], klist[0][1], klist[0][2], klist[0][3]

    def clonomous_check_sim(self, listofmove):
            
            """"for simulation"""
            #temppiece = copy.deepcopy(piece)
            #tempgame = copy.deepcopy(self.game)
            tempboard = copy.deepcopy(self.game.board)
            #tempboard.move(temppiece, move, simulation=True) # move virtually one piece
            print(listofmove)
            for move in listofmove:
                #move = listofmove[0]

                source_row = 7 - move[0][1]
                source_col = move[0][0]
                target_row = 7 - move[1][1]
                target_col = move[1][0]

                if tempboard.squares[source_row][source_col].piece_presence():
                    piece = tempboard.squares[source_row][source_col].piece

                    if piece.color == self.game.player_turn:    
                        tempboard.compute_move(piece, source_row, source_col, bool=False)

                        # get the squares for move
                        source = Square(source_row, source_col)
                        target = Square(target_row, target_col)

                        move = Move(source, target)

                        #  check move ok ?

                        if not tempboard.valid_move(piece, move):
                            listofmove.pop(listofmove.index(listofmove[0]))
                            print("%s poped out" % self.clone_chess.convert_move_2_string(move))
                            
                        else:
                            print("find move", self.clone_chess.convert_move_2_string(move))
                            return source_row, source_col, target_row, target_col
                            #self.autonomous_check_sim(piece, listofmove)

    

                        
    
    def autonomous_piece(self,source_row, source_col, target_row, target_col, board, game, clone_chess, surface):
        # presence of piece within selected square
        
        if self.game.board.squares[source_row][source_col].piece_presence():
            piece = self.game.board.squares[source_row][source_col].piece

            if piece.color == self.game.player_turn:

                board.compute_move(piece, source_row, source_col, bool=True)

                # get the squares for move
                source = Square(source_row, source_col)
                target = Square(target_row, target_col)

                move = Move(source, target)

                #  check move ok ?
                if game.board.valid_move(piece, move):
                    print("AUTONO OK MOVE ZONE--")
                    board.move(piece, move)

                    if piece.type == chess.PAWN and game.board.squares[target_row][target_col].piece.type == chess.QUEEN:

                         # BRIDGE HERE cloning move from app to python-chess
                        self.pawn_promotion(source_row, source_col, target_row, target_col)

                    else:
                            # BRIDGE HERE cloning move from app to python-chess"""
                        clone_chess.move_clone_board(move)

                    board.set_true_en_passant(piece)

                    print(piece.name, "AUTONOMOUS from",source_col, 7-source_row,"to",target_col, 7-target_row)
                    
                    # uncomment to get FEN output
                    #print("AUTONOMOUS FEN output: ",clone_chess.get_fen())

                    game.display_chessboard(surface)
                    game.display_lastmove(surface)
                    game.display_pieces(surface)
                    # print the Outcome of the game
                    board.check_termination(clone_chess.get_board())

                    game.next_player()
                
                else:
                    print("DEAD GAME ZONE--")
                    board.check_termination(clone_chess.get_board())
                    game.reset()

        else:
            print("No piece")

main = Main()
main.mainloop()