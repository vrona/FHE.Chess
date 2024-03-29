# Inference

Documents for all 3 different scripts related to 3 modes:<br>
- clear: [infer_clear.py](../server_cloud/server/infer_clear.py)
- simfhe: [infer_simfhe.py](../server_cloud/server/infer_simfhe.py)
- deepfhe: [infer_deepfhe.py](../server_cloud/server/infer_deepfhe.py)

### Predict method

```predict(input_board, topf=2, topt=3)``` method from ```Inference``` class operates with some similarities and differences.<br>

- **Choice of number of outputs**<br>
```input_board```, ```topf=2```, ```topt=3``` parameters are respectively the current Python-Chess lib's ```chess.Board()```, the top (highest) 2 values wanted in the (64,) Source's output and, for each one, the top 3 values wanted in the (64,) Target's output.<br>
(**recall NB**: as we want a square number as final inferred data, we use the index of the top scores). This tops are retrieved differently when in different contexts.<br>


    - **clear**

    Outputs are PyTorch's tensor type, then ```torch.topk``` is solicited.

    ```python
    # 2 topf source square
    _, source_square = torch.topk(source_output, topf)

    #...
    
    # topt target square
    _, target_square = torch.topk(target_output, topt)
    
    #...
    ```

    - **simfhe and deepfhe**

    Outputs are Numpy array type, then ```np.argsort``` is used followed by data manipulations.
    

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
    <br>

    - **clear**
    
    They are loaded in the script.

    ```python
    # loading the checkpoint
    source_state_dict = torch.load("weights/source_clear.pt",map_location = device)
    target_state_dict = torch.load("weights/target_clear.pt",map_location = device)
    ```
    An alternative would have to instantiate them as class attributes.<br>
    <br>
    
    - **simfhe**
    

    They are surely instantiated as class attributes (```self.source_model, self.target_model```) and recall that fhe simulation needs them to be compiled first.<br>
    This step should be repeated everytime but as the whole application is built on client-server architecture, when the Server is initializing and connecting with the Client, **compilation happens only once**.<br>

    The trick is done when running either [server_all.py](../server_cloud/server/server_all.py) or [server_simfhe.py](../server_cloud/server/server_simfhe.py). For example in the case of server_simfhe.py, it happens precisely here:<br>
    
    ```python
    # from compile_fhe_inprod import CompileModel
    compiled_models = CompileModel()
    inference = Inference_simfhe(compiled_models.compiled_source, compiled_models.compiled_target)
    ```
    <br>

    - **deepfhe**:
    
    This step is handled specifically by [deep_fhe.py](../server_cloud/client/deep_fhe.py).<br>

    As client is for encryption and decryption with private keys, Source and Target models' cryptographic components are loaded by ```FHEModelClient``` method (imported from Concrete-ML).<br>
    And as server infers with evaluation public keys, thus models' cryptographic components are loaded by ```FHEModelServer```.<br>

    ```python
    self.source_client = "client/source"
    self.target_client = "client/target"
    
    # source
    self.fhesource_client = FHEModelClient(self.source_client, key_dir=self.source_client)
    self.fhesource_client.load()
    
    # target
    self.fhetarget_client = FHEModelClient(self.target_client , key_dir=self.target_client)
    self.fhetarget_client.load()
    
    ## server
    self.source_server = "server/model/source"
    self.target_server = "server/model/target"
    
    # source
    self.fhesource_server = FHEModelServer(path_dir=self.source_server)#, key_dir=self.source_server)
    self.fhesource_server.load()
    
    # target
    self.fhetarget_server = FHEModelServer(path_dir=self.target_server)# , key_dir=self.target_server)
    self.fhetarget_server.load()
    ```

- **Inference**

    Here, the excerpt of Source model's inference.<br>
    Target's one is the same but takes 2 input_data: ```chessboard, source_square_bit```.
    
    - **clear**

    ```python
    source_output = source_model(torch.tensor(board).unsqueeze(0).to(torch.float).to(device))
    ```

    - **simfhe**

    ```python
    # Prediction of source square
    # adding dim + from torch to numpy type
    source_input  = torch.tensor(board).unsqueeze(0).to(torch.float).to(device)
    source_input = source_input.cpu().detach().numpy()

    # zama fhe simulation quantization --> encryptions, keys check --> inference
    source_input_q = source_model.quantize_input(source_input)
    source_pred = source_model.quantized_forward(source_input_q, fhe="simulate")
    
    # dequantization <-- decryptions <-- inference
    source_output = source_model.dequantize_output(source_pred)
    ```

    - **deepfhe**

    ```python
    # Prediction of source square
    # adding dim + from torch to numpy type
    source_input  = torch.tensor(board).unsqueeze(0).to(torch.float).to(device)
    source_input = source_input.cpu().detach().numpy()
    
    # zama fhe for real with FHEModelClient FHEModelServer quantization --> encryptions, keys check --> inference
    source_encrypted, source_keys = self.fhe_chess.encrypt_keys(source_input)
    source_serial_result = self.fhe_chess.fhesource_server.run(source_encrypted, source_keys)

    # dequantization <-- decryptions <-- inference
    source_output = self.fhe_chess.decrypt(source_serial_result)
    ```

    We find here again ```FHEModelClient, FHEModelServer``` through ```FHE_chess```'s attributes (```encrypt_keys(), fhesource_server, decrypt()```) instantiated as ```self.fhe_chess``` from [deep_fhe.py](../server_cloud/client/deep_fhe.py).<br>

    ```python
        """
        input: float --> quantization --> encryption
        """
        def encrypt_keys(self, clear_chessboard, clear_source=None, target=False):
            
            if clear_source is not None and target:
                
                # quantized and encryption of clear target model inputs
                serial_target_evaluation_keys = self.fhetarget_client.get_serialized_evaluation_keys()
                quantized_chessboard, quantized_source = self.fhetarget_client.model.quantize_input(clear_chessboard, clear_source)
                chess_encrypt_input, source_encrypt_input = self.fhetarget_client.quantize_encrypt_serialize(quantized_chessboard, quantized_source)

                return chess_encrypt_input, source_encrypt_input, serial_target_evaluation_keys
            else:

                # quantized and encryption of clear source model input            
                quantized_chessboard = self.fhesource_client.model.quantize_input(clear_chessboard)
                serialized_evaluation_keys = self.fhesource_client.get_serialized_evaluation_keys()
                encrypted_input = self.fhesource_client.quantize_encrypt_serialize(quantized_chessboard)
                
                return encrypted_input, serialized_evaluation_keys
        
        """
        output: float <-- dequantization <-- decryption
        """
        def decrypt(self, encrypted_output, target=False):

            if target:
                decrypted_data = self.fhetarget_client.deserialize_decrypt_dequantize(encrypted_output)
                return decrypted_data
            else:
                decrypted_data = self.fhesource_client.deserialize_decrypt_dequantize(encrypted_output)
                return decrypted_data
    ```
    
    NB: simfhe and deepfhe scripts have an additional piece of code to delete any predicted Source square without a white piece.<br>
    It had been added while testing and left since then.<br>
    
    ```python
    # checking source_square prediction is white pieces, if not deletation
    indices_to_remove = []

    for d in range(topf):

        # thanks to chess lib, provide the #number of the square and return type :P, ...
        if str(input_board.piece_at(source_square[:,d][0])) not in white_pieces:
            indices_to_remove.append(d)

    square_toremove = source_square[0][indices_to_remove]
    source_square = source_square[~np.isin(source_square,square_toremove)]
    # warning source_square.shape from dim=2 to dim=1
    ```
    
    **Finally** <br>
    All the proposals (topt for every topf) are gathered into a dict (after a needed translation operated by ```square_to_alpha()```method, see below) and are then filtered by ```legal_moves``` generator from Python-Chess lib.<br>
    To obtain some more permissive moves ```pseudo_legal_moves``` generator is also involved.<br>
    A list of tuples is returned.
    <br>

    ```python
    # from raw proposal to legal propose
        for i, (prop, values) in enumerate(dictofproposal.items()):
        
            if chess.Move.from_uci(prop) in input_board.legal_moves:
                #print("Move %s -- %s legal" %(i,prop))
                legal_proposal_moves.append(values)
    
            elif chess.Move.from_uci(prop) in input_board.pseudo_legal_moves:
                pseudo_legal_propmoves.append(values)

        if len(legal_proposal_moves)== 0 and len(pseudo_legal_propmoves)>0:
           print("pseudo legal moves available")
           return pseudo_legal_propmoves

        else:
            return legal_proposal_moves

    ```
    
    The list of inferences is exploited by [Chess_app](Chess_app/Chess_app.md) by [main.py](../client_local/chess_env/main.py), here:

    ```python
    listoftuplesofmoves = cs_network.send(chessboard)
    ```

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