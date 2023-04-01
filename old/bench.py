
"""
Evaluation
- Raw material
- Positional information
- Future moves
"""

"""
NN: board -> Piece selection
NN: board position -> eval score
"""

"""
piece's board position + move index + color to move
"""


""" they have two outputs:
- one for evaluation (the value head),
- and one for move ordering (the policy head),
rather than only one output for evaluation.
"""

"""
board.is_check()
board.is_stalemate()"""