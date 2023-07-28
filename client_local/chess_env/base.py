import numpy as np
# screenplay dimension
sp_width = 800
sp_height = 800

# chess board
cb_cols = 8
cb_rows = 8
sqsize = sp_width // cb_cols

# array of square table location within chessboard (8x8) 
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

"""
homemade chessboard row starts at 0 from top // col starts at 0 from left
chess lib chessboard row starts at 1 from bottom // col starts at a from left or bitboard from 0 to 63 squares
"""