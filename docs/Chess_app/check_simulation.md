# Check simulation

check means the King is under attack.

Virtualization of the board by copying the Board() and Piece() classes. Then, compute the possible moves for a given attacker. If the piece in the destination square happen to be a King return True.

<br/>


### [client_local/chess_env/board.py](../client_local/chess_env/board.py)
```python
89         def king_check_sim(self, piece, move):
90             
91             """"for simulation"""
92             temppiece = copy.deepcopy(piece)
93             tempboard = copy.deepcopy(self)
94             tempboard.move(temppiece, move, simulation=True) # move virtually one piece
95     
96             for row in range(cb_rows):
97                 for col in range(cb_cols):
98                     """check for all opponent if their potential ok_moves arrive in the team's Kings' square"""
99                     if tempboard.squares[row][col].opponent_presence(piece.color):
100                        p = tempboard.squares[row][col].piece
101                        
102                        tempboard.compute_move(p, row, col, bool=False)
103    
104                        for mvmt in p.ok_moves:
105                            if isinstance(mvmt.target.piece, King):
106                                return True
107            return False
```

<br/>

<br/>

<br/>