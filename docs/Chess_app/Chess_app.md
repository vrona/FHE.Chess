# Chess App

The Chess environment has been developed from scratch thanks to these tutorials and wiki: [Bibliography](../Biblio.md).
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
[base.py](../../client_local/chess_env/base.py): pro<br>
[board.py](../../client_local/chess_env/board.py)<br>
[button.py](../../client_local/chess_env/button.py)<br>
[clone_chess.py](../../client_local/chess_env/clone_chess.py)<br>
[config.py](../../client_local/chess_env/config.py)<br>
[dragger.py](../../client_local/chess_env/dragger.py)<br>
[game.py](../../client_local/chess_env/game.py)<br>
[main.py](../../client_local/chess_env/main.py)<br>
[move.py](../../client_local/chess_env/move.py)<br>
[piece.py](../../client_local/chess_env/piece.py)<br>
[square.py](../../client_local/chess_env/square.py)<br>