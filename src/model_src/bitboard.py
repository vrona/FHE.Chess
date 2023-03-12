import chess
#rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
board = chess.Board('r1b1kbn1/pppqnpp1/7r/1B1pp2p/3PP3/5P1N/PPPK2PP/RNBQR3 w q - 5 8')

legal_moves = list(board.legal_moves)
WP = board.pieces(chess.PAWN, chess.WHITE)
WR = board.pieces(chess.ROOK, chess.WHITE)
BP = board.pieces(chess.PAWN, chess.BLACK)

BB = board.pieces(chess.BISHOP, chess.BLACK)
BR = board.pieces(chess.ROOK, chess.BLACK)


print (int(WR), int (BR))

print (list(WR),
       list(BR))

#print ("\nWHITE PAWNS:\n" + str(WR))
#print ("\nBLACK PAWNS:\n" + str(BR))

# for i in legal_moves:
#     print(i)