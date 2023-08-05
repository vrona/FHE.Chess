# Model Dev. / Training / Validation / Testing

3 subsets: training_set, valid_set, test_set are made from global dataset [wb_2000_300](../server_cloud/data/wb_2000_300.csv).<br>
```python
# split dataset splitted into: training_set (80%), valid_set (20%), test_set (20%)
training_set, valid_set, test_set = np.split(wechess.sample(frac=1, random_state=42), [int(.6*len(wechess)), int(.8*len(wechess))])
```
They are instantiated for eg.:
```python
# thanks to dataset_source or dataset_target
trainset = Chessset(training_set['AN'], training_set.shape[0])

# loaded with Dataloader (PyTorch method) where shuffle game and batch size parameters are specified.
train_loader = DataLoader(trainset, batch_size = 64, shuffle=True, drop_last=True)
```

### **Clear**

*   Training and Testing are managed by running [launch_train_test_clear.py](../server_cloud/traintest_only/launch_train_test_clear.py).<br>

    Clear models are trained, validated, tested on non-encrypted data thanks to [train_source_clear](../server_cloud/traintest_only/train_source_clear.py) and [train_target_clear](../server_cloud/traintest_only/train_target_clear.py).<br>

    Training parameters:<br>
    ```python
    Epochs = 5
    Learning_rate = 1.0e-3
    criterion = nn.MSELoss()
    ```

    ```train_loss```, ```valid_loss``` and ```accuracy``` are monitored by wandb (aka [Weights & Biases](https://wandb.ai/site)).<br>

    **NB**: the level of float precision offered by ```torch.float``` is enough.<br>

*   Models

    *   Source: [cnn_source_clear](../server_cloud/model_src/clear/cnn_source_clear.py)
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

    *   Target: [cnn_target_clear](../server_cloud/model_src/clear/cnn_target_clear.py)



### **Quantization**

Quantized model (clear models are converted into an integer equivalent) trained, validated, tested on non-encrypted data.

*   Run training & testing: 

*   Models

    *   Source: [source_44cnn_quantz.py](../server_cloud/model_src/quantz/source_44cnn_quantz.py)

    *   Target (**training**): [target_44cnn_quantz.py](../server_cloud/model_src/quantz/target_44cnn_quantz.py)

    *   Target (**inference**): [target_44cnn_quantz_eval.py](../server_cloud/model_src/quantz/target_44cnn_quantz_eval.py)



### **Compilation & Simulation** (Virtual Library)

*   Run test (model are compiled with Concrete's FHE compiler to run inference on encrypted data): blob/quant_fhe/server_cloud/traintest_only/launch_(test)_compile_fhe.py)

## Model Deployment

*   client generates private keys and a public evaluation key (used by the model's FHE evaluation on the server) and then encrypts data and decrypts results.

*   server runs compiled model, makes inference on encrypted data.

<br/>