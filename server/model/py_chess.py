import chess
import chess.pgn

pgn = open('/Volumes/vrona_SSD/FHE.Chess/docs/kasparov-deep-blue-1997.pgn')
first_game = chess.pgn.read_game(pgn)
#chessboard = chess.Board()
board = first_game.board()
for move in first_game.mainline_moves():
    board.push(move)

board
#print(board)