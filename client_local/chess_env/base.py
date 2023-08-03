import numpy as np

# screenplay dimension
sp_width = 800
sp_height = 800

"""
RECALL:
homemade chessboard row starts at 0 from top // col starts at 0 from left
python-chess library (https://python-chess.readthedocs.io): within chessboard, rows start at "1" from bottom  // cols start at "a" from left // bitboard from 0 to 63 squares (see below)
"""

# homemade chessboard (8*8)
cb_cols = 8                     #number of columns
cb_rows = 8                     #number of rows
sqsize = sp_width // cb_cols    #size of squares


"""Python-Chess lib. uses alphanumeric coordinates (e2 equivalent to col 1 - row 2) but also
bitboard logic: an array of square table location within chessboard.
the move e2e4 becomes 8 to 24.
"""
bitboard = np.array([
    [56,57,58,59,60,61,62,63],
    [48,49,50,51,52,53,54,55],
    [40,41,42,43,44,45,46,47],
    [32,33,34,35,36,37,38,39],
    [24,25,26,27,28,29,30,31],
    [16,17,18,19,20,21,22,23],
    [8,9,10,11,12,13,14,15],
    [0,1,2,3,4,5,6,7],
    ])