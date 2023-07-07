import chess
from clone_chess import Clone_Chess

mychess = Clone_Chess()

board = chess.Board("rnb1kbnr/pp2pppp/2pp4/q7/6P1/3P4/PPPBPP1P/RN1QKBNR b KQkq - 2 4")
print(board)

print(chess.Move.from_uci("a5d2") in board.legal_moves)
