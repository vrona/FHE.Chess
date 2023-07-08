import chess
from clone_chess import Clone_Chess

# get fen
board_init = chess.Board("rnb1kbnr/pp2pppp/2pp4/q7/6P1/3P4/PPPBPP1P/RN1QKBNR b KQkq - 2 4")
print(board_init)

"""
temp board_new
board_temp = chess.Board()
for each movement:
    board_temp.push_uci("a5d2")
    engine.analysis(board)
    print result['score']

"""
print(chess.Move.from_uci("a5d2") in board_init.legal_moves)