# Data Explanation

PGN (Portable Game Notation) file contains \[annotations\] which pair a Tag with a "String value" and then all the games' moves.
Each move is numbered followed by White's move (Pawn move to e4) then Black's one (Pawn move to e5).
The outcome is notified by symbol "+" (Check), "#" (Checkmate) and "1-0" if White wins otherwise that is Black ("0-1") or Draw ("0-0").

match _pgn_ (portable game notation) data example formatted to be imported into Lichess: [/lichess.org/paste](https://lichess.org/paste)

```text
1      [Event                                                         "Blitz"]
2      [White                                                         "go4jas"]
3      [Black                                                     "Sergei1973"]
4      [Result                                                           "0-1"]
5      [UTCDate                                                   "2016.06.30"]
6      [UTCTime                                                     "22:00:01"]
7      [WhiteElo                                                        "1641"]
8      [BlackElo                                                        "1627"]
9      [WhiteRatingDiff                                                "-11.0"]
10     [BlackRatingDiff                                                 "12.0"]
11     [ECO                                                              "C20"]
12     [Opening                                    "King's Pawn Opening: 2.b3"]
13     [TimeControl                                                    "300+0"]
14     [Termination                                                   "Normal"]
15     1. e4 e5 2. b3 Nf6 3. Bb2 Nc6 4. Nf3 d6 5. d3 g6 6. Nbd2 Bg7 7. g3 Be6 8. Bg2 Qd7 9. O-O O-O 10. c3 b5 11. d4 exd4 12. cxd4 Bg4 13. Rc1 Rfe8 14. Qc2 Nb4 15. Qxc7 Qxc7 16. Rxc7 Nxa2 17. Ra1 Nb4 18. Raxa7 Rxa7 19. Rxa7 Nxe4 20. Nxe4 Rxe4 21. Ng5 Re1+ 22. Bf1 Be2 23. Rxf7 Bxf1 24. Kh1 Bh3# 0-1
```
<br>

Basically, the last line which gathers all the moves. The raw dataset that have been used in this project is composed of juxtaposition of moves of complete games like this.<br> (see [data_transformation](data_transformation.md) and [jupyter notebook](../server_cloud/data/wb_2000.ipynb))
```text
1. e4 e5 2. b3 Nf6 3. Bb2 Nc6 4. Nf3 d6 5. d3 g6 6. Nbd2 Bg7 7. g3 Be6 8. Bg2 Qd7 9. O-O O-O 10. c3 b5 11. d4 exd4 12. cxd4 Bg4 13. Rc1 Rfe8 14. Qc2 Nb4 15. Qxc7 Qxc7 16. Rxc7 Nxa2 17. Ra1 Nb4 18. Raxa7 Rxa7 19. Rxa7 Nxe4 20. Nxe4 Rxe4 21. Ng5 Re1+ 22. Bf1 Be2 23. Rxf7 Bxf1 24. Kh1 Bh3# 0-1
```