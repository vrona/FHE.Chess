# Chess App

The Chess environment has been developed from scratch thanks to these tutorials and wiki: [Bibliography](../Biblio.md).
It integrates [Python-Chess library](https://python-chess.readthedocs.io/en) via the Clone_chess class here: [clone_chess.py](../../client_local/chess_env/clone_chess.py)

```text
├── client_local
│   ├── [chess_env](client_local/chess_env)
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