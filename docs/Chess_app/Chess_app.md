# Chess App
**In this version [Python-Chess library](https://python-chess.readthedocs.io/en) is at the core mechanism of the app.**
The Chess environment has been developed from scratch thanks to the "Coding a Complete Chess Game" tutorial and wiki (see [bibliography](../bibliography.md)).
It integrates [Python-Chess library](https://python-chess.readthedocs.io/en) via the ```Clone_chess``` class (see [clone_chess.py](../../client_local/chess_env/clone_chess.py))

```text
├── client_local
│   ├── chess_env
│   │   ├── base.py
│   │   ├── board.py
│   │   ├── button.py
│   │   ├── clone_chess.py
│   │   ├── config.py
│   │   ├── dragger.py
│   │   ├── game.py
│   │   ├── main.py
│   │   ├── move.py
│   │   ├── piece.py
│   │   └── square.py
│   ├── chess_network.py (handles client (your local machine) - server (remote instance) connection)
│   └── content
│       ├── imgdot (folder of images of dots. Used to show the possibles target squares for the selected piece)
│       └── pieces
│           ├── pieces_128px (folder of all black & white chess pieces. Used when they are been grabbed)
│           └── pieces_80px (folder of all black & white chess pieces. Used when they are on the chessboard)

```
<br>

### Structural<br>

[base.py](../../client_local/chess_env/base.py)<br>
structural information about the chessboard itself and enables interaction between "homemade" chess environment and python-chess library.<br>

[square.py](../../client_local/chess_env/square.py)<br>
A class that mainly returns bool about the content of squares (opponent piece, empty square, ...), the chessboard's limits and alphanumeric conversion.<br>

[piece.py](../../client_local/chess_env/piece.py)<br>
The class that defines what is a piece: name, color, image, behavior on the chessboard, list of authorized moves, ... and then each type of piece is a class on its own. This is much needed because of specific moves that Pawn and King have (respectively promotion, en-passant and rooking -left vs right-).

[move.py](../../client_local/chess_env/move.py)<br>
The class that explicitly defines what composes a move (FROM: source square, TO: target square).

[board.py](../../client_local/chess_env/board.py)<br>
This class used all the 4 previous classes to create methods which define pieces behaviors on the chessboard and basically the creation of piece inside the squares.<br>

There are several methods that are notable:
<br>

- Core methods:
    - defining the pieces' legal moves (moves which avoid to put King in check.):
        ```python
        def piece_legal(self, current_board, piece):
        """
        legal move form python-chess, algebraic format of SOURCE square >> algebraic format of TARGET squares  >  push moves to piece.legal_move list.
        """
        # make a list of legal moves from python-chess
        list_legal = list(current_board.legal_moves)

        # make a dict of legal moves where keys is a tuple of splitted sprint from keys
        coordinate_legal = {tuple(alphasq):[] for alphasq in alphanum_square.keys()}
        
        for i, lv in enumerate(list_legal):
            coordinate_legal[tuple(str(list_legal[i])[:2])].append(tuple(str(lv)[2:]))

        for source_alpha, target_list_tuple in coordinate_legal.items():

            source = Square(8 -int(source_alpha[1]), Square.convert_algeb_not(source_alpha[0]))

            for target_alpha in target_list_tuple: #('g', '1', 'f', '3')
                target = Square(8 -int(target_alpha[1]), Square.convert_algeb_not(target_alpha[0]))
    
                move = Move(source, target)
        
                if self.squares[8 -int(source_alpha[1])][Square.convert_algeb_not(source_alpha[0])].piece_type() is not None: #and current_board.is_check() != True

                    self.squares[8 -int(source_alpha[1])][Square.convert_algeb_not(source_alpha[0])].piece.add_legalmove(move)
        ```

- Common methods:
    - ```python
        def move(piece, move, simulation = False)
        ```
    

- Pieces have exceptional movements:

    - Pawn:
        - Promotion:
            ```python
            def check_pawn_promotion(piece, target)
            ```
        - En-passant:
            ```python
            def set_true_en_passant(piece)
            ```
    - King: see [castling doc](docs/Chess_app/castling_move.md)
        ```python
        def castling(source, target)
        ```
<br>

### Game

[dragger.py](../../client_local/chess_env/dragger.py)<br>
A class that enable to provide information to PyGame when grabbing or releasing pieces on the board.

[game.py](../../client_local/chess_env/game.py)<br>
Likely ````dragger```` class, this allows to display all content via PyGame.

### UX<br>
[config.py](../../client_local/chess_env/config.py): used by PyGame for typography.<br>
[button.py](../../client_local/chess_env/button.py): activate AI vs Human or Human vs Human modes.<br>

### Python-Chess power
[clone_chess.py](../../client_local/chess_env/clone_chess.py)<br>
This class calls the Python-Chess methods (see [bibliography](../bibliography.md)).<br>
It is used to clone all piece's location and movements from "homemade" chessboard into Python-Chess module.<br>
This is a key pillar class as AI is nurtured with chessboard data from Python-Chess and its inferred output is filtered with Python-Chess' methods: ```legal_move()``` and ```pseudo_legal_move()```.

### Client-Server
[chess_network.py](../../client_local/chess_network.py) provides the ```Network``` class which takes care of connecting Client (Chess App on local machine) and Server (AI with/without FHE dedicated client-server architecture on a remote instance).

### Main
Everything comes together in [main.py](../../client_local/chess_env/main.py)<br>
<br>


In AI mode, the flow is as follows:
```base```, ```squares```, ```pieces```, ```move``` and ```board``` are initialized (and runs permanently) and are displayed via ```game```.

The input_data (current chessboard) are retrieved from ```clone_chess```:

```python
chessboard = clone_chess.get_board()
```

Then, ```network``` sends them to the Server (AI) and makes its prediction which are sent back to the Client (Chess App). The coordinates of the 1st move are instantiated (which is has been filtered via ```legal_move()``` or ```pseudo_legal_move()``` method).
```python
listoftuplesofmoves = cs_network.send(chessboard)

selected_square_row = listoftuplesofmoves[0][0][1]
selected_square_col = listoftuplesofmoves[0][0][0]
targeted_square_row = listoftuplesofmoves[0][1][1]
targeted_square_col = listoftuplesofmoves[0][1][0]
```

The move is applied to the environment thanks to this method:

```python
self.autonomous_piece(7-selected_square_row, selected_square_col, 7-targeted_square_row, targeted_square_col, board, game, clone_chess, screenplay)
```

**NB**:
- local terminal prints the history of games' moves (chessboard string matrix included) made by either by AI or Human.<br>
    If Forsyth–Edwards Notation (FEN) game position is needed, to print them, just uncomment these lines: 
    - ```python
        #print("\nHUMAN FEN: ",clone_chess.get_fen())
        ```
    - ```python
        #print("\nAUTONOMOUS FEN: ",clone_chess.get_fen())
        ```
- remote terminal prints the chessboard input_data and predictions as a list of tuples.



