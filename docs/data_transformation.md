# Data transformation

At this step, [helper_chessset](../server_cloud/model_src/helper_chessset.py) class makes the action.

## Goal

This document would help the models to receive the desired input_data in correct format.<br>
The models are Convolution Neural Network which need input_data with (12,8,8) format and binary data.<br>

Little take away:<br>
* format :
    - dim 0 = a layer for each color (2) and type of pieces (6),
    - dim 1 = number of cols,
    - dim 2 = number of rows.
* binary :
    - 0: empty square
    - 1: presence of white piece
    - -1: presence of black piece

## Recall

Ready to use "dataset": [wb_2000_300.csv](../server_cloud/data/wb_2000_300.csv) have been made due to data preparation (see. ["wb_2000" jupyter notebook](../server_cloud/data/wb_2000.ipynb)).

## Flow

*   Data (sequences of white vs black which represents a whole game) from "dataset": [wb_2000_300.csv](../server_cloud/data/wb_2000_300.csv):
    ```text
    1. e4 e5 2. b3 Nf6 3. Bb2 Nc6 4. Nf3 d6 5. d3 g6 6. Nbd2 Bg7 7. g3 Be6 8. Bg2 Qd7 9. O-O O-O 10. c3 b5 11. d4 exd4 12. cxd4 Bg4 13. Rc1 Rfe8 14. Qc2 Nb4 15. Qxc7 Qxc7 16. Rxc7 Nxa2 17. Ra1 Nb4 18. Raxa7 Rxa7 19. Rxa7 Nxe4 20. Nxe4 Rxe4 21. Ng5 Re1+ 22. Bf1 Be2 23. Rxf7 Bxf1 24. Kh1 Bh3# 0-1
    ```

*   Here move from this sequence, which is a full game, can be visualized thanks to [Python-Chess library](https://python-chess.readthedocs.io/en/) library: 

```
r . b q k b . r
p p p p . Q p p
. . n . . n . .
. . . . p . . .
. . B . P . . .
. . . . . . . .
P P P P . P P P
R N B . K . N R
```

## Following the flow

<br/>



<br/>

{{Keep adding snippets from the next steps of the flow}}

## Things to note

{{Who uses this flow and when?}}

<br/>