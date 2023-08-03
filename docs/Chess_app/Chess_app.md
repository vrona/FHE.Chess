# Chess App

The Chess environment has been developed from scratch thanks to the "Coding a Complete Chess Game" tutorial and wiki: [Bibliography](../Biblio.md).
It integrates [Python-Chess library](https://python-chess.readthedocs.io/en) via the Clone_chess class here: [clone_chess.py](../../client_local/chess_env/clone_chess.py)

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

Focus on the [Chess_env](../../client_local/chess_env) scripts<br>
<br>
### Structural<br>

[base.py](../../client_local/chess_env/base.py)<br>
structural information about the chessboard itself and enables interaction between "homemade" chess environment and python-chess library.<br>

[square.py](../../client_local/chess_env/square.py)<br>
A class that mainly returns bool about the content of squares (opponent piece, empty square, ...), the chessboard's limits and alphanumeric conversion.<br>

[piece.py](../../client_local/chess_env/piece.py)<br>
The class that defines what is a piece: name, color, image, behavior on the chessboard, list of authorized moves, ... and then each type of piece is a class its own. This is much needed because of specific moves that Pawn and King have. respectively promotion, en-passant and rooking (left vs right).

[move.py](../../client_local/chess_env/move.py)<br>
The class that explicitly defines what composes a move (FROM: source square, TO:target square).

[board.py](../../client_local/chess_env/board.py)<br>
This class used all the 4 previous classes to create methods which define pieces behaviors on the chessboard and basically the creation of piece inside the squares.<br>

Core methods:
- ```compute_move(piece, row, col, bool=True)```
- ```king_check_sim(piece, move)```, (detailed here [check_simulation](Chess_app.md))
- Common methods:<br>
    - ```move(piece, move, simulation = False)```
    - for refactoring purpose: ```move_kingchecksim()``` and ```sim_kingcheck_okmoves()```

Recall that each piece has its own behavior (some shared behavior):

- Pawn:
    - ```check_pawn_promotion(piece, target)```
    - ```set_true_en_passant(piece)```
- King: ```castling(source, target)```

A piece behavior is define by a dedicated internal method inside ```compute_move(piece, row, col, bool=True)``` method.<br>
For eg.: King behavior is defined by ```king_moves()```.

<br>

### Game

[dragger.py](../../client_local/chess_env/dragger.py)<br>
A class that enable to provide information to PyGame when grabbing or releasing pieces on the board.

[game.py](../../client_local/chess_env/game.py)<br>
Likely ````dragger```` class, this allows to display all content via PyGame.

### UX<br>
[config.py](../../client_local/chess_env/config.py)<br>: used by PyGame for typography.
[button.py](../../client_local/chess_env/button.py)<br>: activate and command the game.

### Python-Chess power
[clone_chess.py](../../client_local/chess_env/clone_chess.py)<br>
This class calls the python-chess methods (see [Biblio](../Biblio.md)).<br>
It is used to clone all piece's location and movements from "homemade" chessboard inside python-chess module.<br>
This is a key pillar class as AI is nurtured with data from python-chess and its inferred output is filtered with legal_move() python-chess' method

### Main aka app.
[main.py](../../client_local/chess_env/main.py)<br>






