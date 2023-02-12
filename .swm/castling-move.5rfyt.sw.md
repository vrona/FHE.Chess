---
id: 5rfyt
title: Castling move
file_version: 1.1.2
app_version: 1.2.0
---

Chess rules: one time possibility that two pieces move at once and over another piece (basically only knights can).

At _Kingside_, the King moves two squares to the right while the right Rook moves two squares over the King.

At _Queenside_, the King moves two squares to the left while the right Rook moves three squares over the King. (prerequisites: no pieces must be between the King and the chosen Rook and none of them have moved yet).

<br/>


<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 chessboard/board.py
```python
313                # castling
314                if not piece.moved:
315                    
316                    # queenside
317                    left_rook = self.squares[row][0].piece
318    
319                    if isinstance(left_rook, Rook):
320                        if not left_rook.moved:
321                            for c in range(1, 4):
322                                if self.squares[row][c].piece_presence(): # castling abord because of piece presence
323                                    break
324    
325                                if c == 3:
326                                    piece.left_rook = left_rook # adds left rook to queen
327                                    
328                                    # rook move to king
329                                    # micro location
330                                    initial = Square(row, 0)
331                                    destination = Square(row, 3)
332    
333                                    # move at micro
334                                    rook_move = Move(initial, destination)
335                                    
336                                    # king move to rook
337                                    # micro location
338                                    initial = Square(row, col)
339                                    destination = Square(row, 2)
340    
341                                    # move at micro
342                                    king_move = Move(initial, destination)
343                                    
344                                    if bool:
345                                        if not self.check_simulation(left_rook, rook_move) and not self.check_simulation(piece, king_move): # if not in check go ahead
346                                            left_rook.add_ok_move(rook_move)
347                                            piece.add_ok_move(king_move)
348                                    else:
349                                            left_rook.add_ok_move(rook_move)
350                                            piece.add_ok_move(king_move) # if not in check go ahead
351    
352    
353                    # kingside
354                    right_rook = self.squares[row][7].piece
355    
356                    if isinstance(right_rook, Rook):
357                        if not right_rook.moved:
358                            for c in range(5, 7):
359                                if self.squares[row][c].piece_presence(): # castling abord because of piece presence
360                                    break
361    
362                                if c == 6:
363                                    piece.right_rook = right_rook # adds right rook to king
364                                    
365                                    # rook move to king
366                                        # micro location
367                                    initial = Square(row, 7)
368                                    destination = Square(row, 5)
369    
370                                        # move at micro
371                                    rook_move = Move(initial, destination)
372                                    
373                                    # king move to rook
374                                        # micro location
375                                    initial = Square(row, col)
376                                    destination = Square(row, 6)
377                                    
378                                        # move at micro
379                                    king_move = Move(initial, destination)
380                                    
381                                    if bool:
382                                        if not self.check_simulation(right_rook, rook_move) and not self.check_simulation(piece, king_move): # if not in check go ahead
383                                            right_rook.add_ok_move(rook_move)
384                                            piece.add_ok_move(king_move)
385                                    else:
386                                            right_rook.add_ok_move(rook_move)
387                                            piece.add_ok_move(king_move) # if not in check go ahead
388    
389    
390            if isinstance(piece, Pawn): pawn_moves()
391    
392            elif isinstance(piece, Knight): kight_moves()
393    
394            elif isinstance(piece, Bishop):
395                straightline_move([
396                    (-1,1), #to NE
397                    (-1,-1),#to NW
398                    (1,-1), #to SW
399                    (1,1)   #to SE
400                ])
401    
```

<br/>

This file was generated by Swimm. [Click here to view it in the app](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBRkhFLkNoZXNzJTNBJTNBdnJvbmE=/docs/5rfyt).
