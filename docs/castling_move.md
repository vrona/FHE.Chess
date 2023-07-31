# Castling move
Chess rules: one time possibility that two pieces move at once and over another piece (basically only knights can).

At _Kingside_, the King moves two squares to the right while the right Rook moves two squares over the King.

At _Queenside_, the King moves two squares to the left while the right Rook moves three squares over the King. (prerequisites: no pieces must be between the King and the chosen Rook and none of them have moved yet).


### [client_local/chess_env/board.py](../client_local/chess_env/board.py)
```python
293                # castling
294                if not piece.moved:
295                    
296                    # queenside
297                    left_rook = self.squares[row][0].piece
298    
299                    if isinstance(left_rook, Rook):
300                        if not left_rook.moved:
301                            for c in range(1, 4):
302                                if self.squares[row][c].piece_presence(): # castling abord because of piece presence
303                                    break
304    
305                                if c == 3:
306                                    piece.left_rook = left_rook # adds left rook to queen
307                                    
308                                    # rook move to king
309                                    # micro location
310                                    source = Square(row, 0)
311                                    target = Square(row, 3)
312    
313                                    # move at micro
314                                    rook_move = Move(source, target)
315                                    
316                                    # king move to rook
317                                    # micro location
318                                    source = Square(row, col)
319                                    target = Square(row, 2)
320    
321                                    # move at micro
322                                    king_move = Move(source, target)
323                                    
324                                    if bool:
325                                        if not self.king_check_sim(left_rook, rook_move) and not self.king_check_sim(piece, king_move): # if not in check go ahead
326                                            left_rook.add_ok_move(rook_move)
327                                            piece.add_ok_move(king_move)
328                                    else:
329                                            left_rook.add_ok_move(rook_move)
330                                            piece.add_ok_move(king_move) # if not in check go ahead
331    
332    
333                    # kingside
334                    right_rook = self.squares[row][7].piece
335    
336                    if isinstance(right_rook, Rook):
337                        if not right_rook.moved:
338                            for c in range(5, 7):
339                                if self.squares[row][c].piece_presence(): # castling abord because of piece presence
340                                    break
341    
342                                if c == 6:
343                                    piece.right_rook = right_rook # adds right rook to king
344                                    
345                                    # rook move to king
346                                        # micro location
347                                    source = Square(row, 7)
348                                    target = Square(row, 5)
349    
350                                        # move at micro
351                                    rook_move = Move(source, target)
352                                    
353                                    # king move to rook
354                                        # micro location
355                                    source = Square(row, col)
356                                    target = Square(row, 6)
357                                    
358                                        # move at micro
359                                    king_move = Move(source, target)
360                                    
361                                    if bool:
362                                        if not self.king_check_sim(right_rook, rook_move) and not self.king_check_sim(piece, king_move): # if not in check go ahead
363                                            right_rook.add_ok_move(rook_move)
364                                            piece.add_ok_move(king_move)
365                                    else:
366                                            right_rook.add_ok_move(rook_move)
367                                            piece.add_ok_move(king_move) # if not in check go ahead
```