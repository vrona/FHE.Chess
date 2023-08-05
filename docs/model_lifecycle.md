# Model Dev. / Training / Validation / Testing

### **Clear**

Clear models are trained, validated, tested on non-encrypted data thanks to [launch_train_test_clear.py](../server_cloud/traintest_only/launch_train_test_clear.py).<br>

3 subsets: training_set, valid_set, test_set are made from global dataset [wb_2000_300](../server_cloud/data/wb_2000_300.csv).<br>
```python
# split dataset splitted into: training_set (80%), valid_set (20%), test_set (20%)
training_set, valid_set, test_set = np.split(wechess.sample(frac=1, random_state=42), [int(.6*len(wechess)), int(.8*len(wechess))])
```
They are instantiated like:
```python
trainset = Chessset(training_set['AN'], training_set.shape[0])
```

*   Run training and testing:

*   Models

    *   Source (model 1): 

    *   Target (model 2): 

*   Train, validation, test

    *   Source (model 1): 

    *   Target (model 2): 

### **Quantization**

Quantized model (clear models are converted into an integer equivalent) trained, validated, tested on non-encrypted data.

*   Run training & testing: 

*   Models

    *   Source (model 1): 

    *   Target (model 2): 

*   Train, validation, test

    *   Source (model 1): 

    *   Target (model 2): 

### **Compilation & Simulation** (Virtual Library)

*   Run test (model are compiled with Concrete's FHE compiler to run inference on encrypted data): blob/quant_fhe/server_cloud/traintest_only/launch_(test)_compile_fhe.py)

## Model Deployment

*   client generates private keys and a public evaluation key (used by the model's FHE evaluation on the server) and then encrypts data and decrypts results.

*   server runs compiled model, makes inference on encrypted data.

<br/>