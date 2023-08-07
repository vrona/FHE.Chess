# Inference

3 different scripts for 3 modes:<br>
- clear: [infer_clear.py](../server_cloud/server/infer_clear.py)
- simfhe: [infer_simfhe.py](../server_cloud/server/infer_simfhe.py)
- deepfhe: [infer_deepfhe.py](../server_cloud/server/infer_deepfhe.py)

<br>

### Predict

```predict(input_board, topf=2, topt=3)``` method from ```Inference``` class operates with some similarities and differences.<br>


- **Multiple inferences**:<br>
```input_board```, ```topf=2```, ```topt=3``` parameters are the current Python-Chess lib's ```chess.Board()```, the top (highest) 2 values wanted in the (64,) Source's output and, for each one, the top 3 values wanted in the (64,) Target's output.<br>
(**recall NB**: as we want a square number as final inferred data, we use the index of the top scores). This tops are retrieved differently when in different contexts.<br>


    - **clear**

    Data are PyTorch's tensor type, then ```torch.topk``` is solicited.

    ```python
    # 2 topf source square
    _, source_square = torch.topk(source_output, topf)

    #...
    
    # topt target square
    _, target_square = torch.topk(target_output, topt)
    
    #...
    ```

    - **simfhe and deepfhe**

    Data are Numpy array type, then ```np.argsort``` is used followed by data manipulations.
    

    ```python
    # topf source square
    source_squares = np.argsort(source_output)

    source_square = source_squares[:,-topf:] # getting the indices of the top values but needs to flip them
    source_square = np.flip(source_square) # re-sorted to match source_squares values
    
    #...
    
    # topt target square
    target_squares = np.argsort(target_output)

    target_square = target_squares[:,-topt:] # getting the indices of the top values but needs to flip them
    target_square = np.flip(target_square) # re-sorted to match source_squares values
    
    #...
    ```
            



- **Models' loading**

    - **clear**
    
    <br>
    They are loaded in the script.

    ```python
    # loading the checkpoint
    source_state_dict = torch.load("weights/source_clear.pt",map_location = device)
    target_state_dict = torch.load("weights/target_clear.pt",map_location = device)
    ```
    An alternative would have to instantiate them as class attributes.<br>

    - **simfhe**
    
    <br>

    They are surely instantiated as class attributes (```self.source_model, self.target_model```) and recall that fhe simulation needs them to be compiled first.<br>
    This step should be repeated everytime but as the whole application is built on client-server architecture, when the Server is initialized and connecting with the Client, **compilation happens only once**.<br>

    The trick is done when running either [server_all.py](../server_cloud/server/server_all.py) or [server_simfhe.py](../server_cloud/server/server_simfhe.py). For example in the case of server_simfhe.py, it happens precisely here:<br>
    
    ```python
    # from compile_fhe_inprod import CompileModel
    compiled_models = CompileModel()
    inference = Inference_simfhe(compiled_models.compiled_source, compiled_models.compiled_target)
    ```

    - **deepfhe**:
    
    This step is handled specifically [deep_fhe.py](../server_cloud/client/deep_fhe.py)


### Models' outputs translation


All 3 scripts use ```square_to_alpha(src_sq, trgt_sq)``` method which translates the square number of Source and Target to alphanumeric and digit coordinates.<br>
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