# Inference

3 different scripts for 3 modes:<br>
- clear: [infer_clear.py](../server_cloud/server/infer_clear.py)
- simfhe: [infer_simfhe.py](../server_cloud/server/infer_simfhe.py)
- deepfhe: [infer_deepfhe.py](../server_cloud/server/infer_deepfhe.py)

<br>

They all use ```def square_to_alpha(src_sq, trgt_sq)``` which translate the square number of Source and Target to alphanumeric and digit coordinates.<br>
The first is push into Python-Chess lib's "chess.Board()", the latter into "homemade" board.<br>

```python
def square_to_alpha(self, src_sq, trgt_sq):
       
    """
    convert square number into chessboard digit coordinates (due to chess lib) and alpha.
    input: source square number, target square number
    return : uci format (str), source and target as digit coordinates
    """

    # digit conversion
    col_s, row_s = chess.square_file(src_sq),chess.square_rank(src_sq)
    col_t, row_t = chess.square_file(trgt_sq),chess.square_rank(trgt_sq)

    # alpha conversion
    alpha_col_s = chess.FILE_NAMES[col_s]
    alpha_col_t = chess.FILE_NAMES[col_t]
    
    # converting coordinates to str and join to get uci move format (see chess lib)
    move_proposal = "".join((str(alpha_col_s),str(row_s+1),str(alpha_col_t),str(row_t+1)))

    return  move_proposal, ((col_s,row_s),(col_t,row_t))
```