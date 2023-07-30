# Model Development

### **Clear**

Clear models trained, validated, tested on non-encrypted data.

*   Run training and testing: [server_cloud/traintest\_only/launch\_train\_test\_clear.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/traintest_only/launch_train_test_clear.py)

*   Models

    *   Source (model 1): [server_cloud/model\_src/clear/cnn\_source\_clear.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/model_src/clear/cnn_source_clear.py)

    *   Target (model 2): [server_cloud/model\_src/clear/cnn\_target\_clear.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/model_src/clear/cnn_target_clear.py)

*   Train, validation, test

    *   Source (model 1): [server_cloud/traintest\_only/train\_source\_clear.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/traintest_only/train_source_clear.py)

    *   Target (model 2): [server_cloud/traintest\_only/train\_target\_clear.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/traintest_only/train_target_clear.py)

### **Quantization**

Quantized model (clear models are converted into an integer equivalent) trained, validated, tested on non-encrypted data.

*   Run training & testing: [server_cloud/traintest\_only/launch\_train\_quantz.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/traintest_only/launch_train_quantz.py)

*   Models

    *   Source (model 1): [server_cloud/model\_src/quantz/source\_44cnn\_quantz.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/model_src/quantz/source_44cnn_quantz.py)

    *   Target (model 2): [server_cloud/model\_src/quantz/target\_44cnn\_quantz.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/model_src/quantz/target_44cnn_quantz.py)

*   Train, validation, test

    *   Source (model 1): [server_cloud/traintest\_only/train\_source\_clear.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/traintest_only/train_source_clear.py)

    *   Target (model 2): [server_cloud/traintest\_only/train\_target\_clear.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/traintest_only/train_target_clear.py)

### **Compilation & Simulation** (Virtual Library)

*   Run test (model are compiled with Concrete's FHE compiler to run inference on encrypted data): [server_cloud/traintest\_only/launch\_(test)\_compile\_fhe.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/traintest_only/launch_(test)_compile_fhe.py)

## Model Deployment

*   client generates private keys and a public evaluation key (used by the model's FHE evaluation on the server) and then encrypts data and decrypts results.

*   server runs compiled model, makes inference on encrypted data.

<br/>