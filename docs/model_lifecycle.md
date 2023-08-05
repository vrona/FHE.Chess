# Model Dev. / Training / Validation / Testing

3 subsets: training_set, valid_set, test_set are made from global dataset [wb_2000_300](../server_cloud/data/wb_2000_300.csv).<br>
```python
# split dataset splitted into: training_set (80%), valid_set (20%), test_set (20%)
training_set, valid_set, test_set = np.split(wechess.sample(frac=1, random_state=42), [int(.6*len(wechess)), int(.8*len(wechess))])
```
They are instantiated, for eg., as follows:
```python
# thanks to dataset_source or dataset_target
trainset = Chessset(training_set['AN'], training_set.shape[0])

# loaded with Dataloader (PyTorch method) where shuffle game and batch size parameters are specified.
train_loader = DataLoader(trainset, batch_size = 64, shuffle=True, drop_last=True)
```

### **Clear**

*   Training, validation and Testing are managed by running [launch_train_test_clear.py](../server_cloud/traintest_only/launch_train_test_clear.py).<br>

    Clear models are trained, validated, tested on non-encrypted data thanks to [train_source_clear.py](../server_cloud/traintest_only/train_source_clear.py) and [train_target_clear.py](../server_cloud/traintest_only/train_target_clear.py).<br>

    Training parameters:<br>
    ```python
    Epochs = 5
    Learning_rate = 1.0e-3
    criterion = nn.MSELoss()
    ```

    ```train_loss```, ```valid_loss``` and ```accuracy``` are monitored by wandb (aka [Weights & Biases](https://wandb.ai/site)).<br>

    **NB**: the level of float precision offered by ```torch.float``` is enough.<br>


*   Models

    *   Source: [cnn_source_clear.py](../server_cloud/model_src/clear/cnn_source_clear.py)
    ```python
    # input_layer, recall 12 input layers is for each 6 types of pieces for each color (2). The output layers is settled at 128 neurons.
    self.input_layer = nn.Conv2d(12, hidden_size, kernel_size=3, stride=1, padding=1)
    ```
    followed by 4 additional CNN layers of 128 neurons, organized with ```nn.ModuleList()``` torch method. <br>
    Their output is then flatted into ```nn.Linear(64,64)```.
    <br>
    Except the source square output which is obtained thanks to
    ```python
    x_source = torch.sigmoid(self.output_source(x))
    ```
    all outputs are resulted from normalization and ```relu``` activation.
    
    <br>

    *   Target: [cnn_target_clear.py](../server_cloud/model_src/clear/cnn_target_clear.py)<br>
       
    Is identical to Source model except that **2 input_data are combined**.<br>
    
    The reason is that the 2nd input_data put emphasis to the selected piece which has to move among all the pieces which are on the current chessboard (aka the 1st input_data).<br>

    1st input_data: ```self.input_layer = nn.Conv2d(12, ...)```<br>
    After all the features have been exploited from CNN layers, the output is flatten to match the 1D format of the 2nd input_data.<br>

    2nd input_data:

    ```python
    # source (the selected squares)
    self.input_source = nn.Linear(64,64)
    ```

    Then, the combination is operated simply at tensor level:

    ```python
    # merging chessboard (context + selected source square)
    merge = chessboard + source

    merge = self.batchn1d_1(merge)
    ```

    This is followed by a much needed normalization step. Indeed, ```chessboard``` variable is the 1D (64,) output from flatten CNN layers filled of floats between 0 and 1. ```source```variable is also a 1D (64,) but filled of 0 and just 1 at the indice relative to the square number where is located the piece in the bitboard. Then, for better computation results at this indice in the "merged" tensor, the feature is then below 1 (so do accordingly the other remaining 63 features).


### **Quantization**

Quantized model (clear models are converted into an integer equivalent) trained, validated, tested on non-encrypted data.<br>

At this step, needs a deep dive into Quantization? read [zama's quantization explanations](https://docs.zama.ai/concrete-ml/advanced-topics/quantization)<br>

*   Training, validation and Testing are **identical as Clear except**:<br>

    Training and validation are managed by running [launch_train_quantz.py](../server_cloud/traintest_only/launch_train_quantz.py).<br>

    Testing is managed by running [launch_(test)_compile_fhe.py](../server_cloud/traintest_only/launch_(test)_compile_fhe.py)<br>

    Quantized models are trained, validated, tested on non-encrypted data thanks to [train_source_quantz.py](../server_cloud/traintest_only/train_source_quantz.py) and [train_target_quantz](../server_cloud/traintest_only/train_target_quantz.py).<br>

    Training parameters:<br>
    ```python
    Epochs = 10
    Learning_rate = 1.0e-3
    criterion = nn.MSELoss()
    ```
    Epoch have been doubled to compensate a bit the eventual losses of accuracy due to the action of quantization which reduces the precision of values at tensor .<br>

*   Models
    Both "Quantized" models keep the same structure as "Clear" ones.<br>
    
    All type layer and activations (PyTorch) methods are changed into their "quantized" (Brevitas) equivalent:<br>
        
    -  convolution: ```nn.Conv2d()``` to ```qnn.QuantConv2d()```
    -  linear: ```nn.Linear()``` to ```qnn.QuantLinear()```
    -  activation: ```F.relu()``` to ```qnn.QuantReLU()```
    -  activation: ```torch.sigmoid()``` to ```qnn.QuantSigmoid()```
    
    ```nn.BatchNorm2d, nn.BatchNorm1d``` from PyTorch are kept. They offer better results and the 2nd does not handle dimension properly.

    <br>

    **A neuralgic method must not be forgotten**: ```qnn.QuantIdentity``` before feeding each or groups of layers.<br>
    It sets the ```scale, zero_point, bit_width, signed_t, training_t``` parameters to the followed layers applied to values at tensor level.<br>

    Then, bit_width, weight_width, return_quant_tensor and pruning are the last key elements used in this context.<br>
    - bit_width and weight_width ```n_bits, w_bits``` are the number of bits necessary from input_data and weights for intermediary results. Here, they settled to 4 and 4.
    - ```return_quant_tensor``` is mainly settled to ```True``` to get a quantized output from layer.
    - pruning technic is used to help model to produce intermediary and global outputs with fewer bits as it sets a maximum of neurons to be activated among specific layers. Here, a max of 84 out of 128 are activated among the CNN layers.

    *   Source: [source_44cnn_quantz.py](../server_cloud/model_src/quantz/source_44cnn_quantz.py)

    *   Target (**training**): [target_44cnn_quantz.py](../server_cloud/model_src/quantz/target_44cnn_quantz.py)

    *   Target (**inference**): [target_44cnn_quantz_eval.py](../server_cloud/model_src/quantz/target_44cnn_quantz_eval.py)



### **Compilation & Simulation** (Virtual Library)

*   Run test (model are compiled with Concrete's FHE compiler to run inference on encrypted data): blob/quant_fhe/server_cloud/traintest_only/launch_(test)_compile_fhe.py)

## Model Deployment

*   client generates private keys and a public evaluation key (used by the model's FHE evaluation on the server) and then encrypts data and decrypts results.

*   server runs compiled model, makes inference on encrypted data.

<br/>