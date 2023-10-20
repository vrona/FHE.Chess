# Castling move
Chess rules: one time possibility that two pieces move at once and over another piece (basically only knights can).

At _Kingside_, the King moves two squares to the right while the right Rook moves two squares over the King.

At _Queenside_, the King moves two squares to the left while the right Rook moves three squares over the King. (prerequisites: no pieces must be between the King and the chosen Rook and none of them have moved yet).


### [/chess_env/board.py](../../client_local/chess_env/board.py)
```python
        def king_moves():

            # castling
            if not piece.moved:
                
                # queenside
                left_rook = self.squares[row][0].piece

                if isinstance(left_rook, Rook):
                    if not left_rook.moved:
                        for c in range(1, 4):
                            if self.squares[row][c].piece_presence(): # castling abord because of piece presence
                                break

                            if c == 3:
                                piece.left_rook = left_rook # adds left rook to queen
                                
                                # rook move to king
                                # micro location
                                source = Square(row, 0)
                                target = Square(row, 3)

                                # move at micro
                                rook_move = Move(source, target)
                                
                                # king move to rook
                                # micro location
                                source = Square(row, col)
                                target = Square(row, 2)

                                # move at micro                                
                                left_rook.add_legalmove(rook_move)


                # kingside
                right_rook = self.squares[row][7].piece

                if isinstance(right_rook, Rook):
                    if not right_rook.moved:
                        for c in range(5, 7):
                            if self.squares[row][c].piece_presence(): # castling abord because of piece presence
                                break

                            if c == 6:
                                piece.right_rook = right_rook # adds right rook to king
                                
                                # rook move to king
                                    # micro location
                                source = Square(row, 7)
                                target = Square(row, 5)

                                    # move at micro
                                rook_move = Move(source, target)
                                
                                # king move to rook
                                    # micro location
                                source = Square(row, col)
                                target = Square(row, 6)
                                
                                    # move at micro
                                right_rook.add_legalmove(rook_move)
```