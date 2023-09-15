# Model Deployment

This documentation is about [client_server_fhe_deploy.py](../server_cloud/client_server_fhe_deploy.py) scripts.<br>

### Principle

Exporting the Source and Target models to their respective client-server locations (as below) and generating cryptographic elements:<br>

- **client (local machine)**:
    - chess code
    - network code.
- **server (remote machine)**:
    - client:
        - source: private keys generated for encryption (input_data: chessboard) and decryption (output: source square)
        - target: private keys generated for encryption (input_data_1: chessboard + input_data_2: source square) and decryption (output: target square)
    - server:
        - model:
            - source: public evaluation keys for inference on encrypted input_data
            - target: public evaluation keys for inference on encrypted input_data_1 + input_data_2

<br>


### Coding

After the both models have been loaded (with pruning unenabled) and compiled with hundreds of data point.<br>

A ```class OnDiskNetwork``` helps to save, copy models and send them to client to generate private keys. Then, sends them to server and producing evaluation keys.<br>

It has been written by Zama's team and can be found in this [ClientServer notebook](https://github.com/zama-ai/concrete-ml/blob/release/1.1.x/docs/advanced_examples/ClientServer.ipynb) and in other [deployment test cases](https://github.com/zama-ai/concrete-ml/tree/9096a9d4f106b486532ec77a26a2cb8e423ebcf1/tests/deployment).<br>


```python
"""
🅝🅔🅣🅦🅞🅡🅚/🅢🅐🅥🅘🅝🅖/🅢🅔🅡🅥🅔🅡 🅢🅔🅒🅣🅘🅞🅝
cf. zama documentation
"""
# instantiating the network
network = OnDiskNetwork()

source_dev = network.dev_dir+"/source"
target_dev = network.dev_dir+"/target"

# saving trained-compiled model and sending to server
## model source
### Now that the model has been trained we want to save it to send it to a server
fhemodel_src = FHEModelDev(source_dev, q_model_source)
fhemodel_src.save()
print("model_source_saved")

### sending models to the server
network.dev_send_model_to_server("/source")
print("model_source_senttoserver")

## model target
fhemodel_trgt = FHEModelDev(target_dev, q_model_target)
fhemodel_trgt.save()
print("model_target_saved")

network.dev_send_model_to_server("/target")
print("model_target_senttoserver")

# send client specifications and evaluation key to the client
network.dev_send_clientspecs_and_modelspecs_to_client("/source")
network.dev_send_clientspecs_and_modelspecs_to_client("/target")

"""
🅒🅛🅘🅔🅝🅣
cf. zama documentation
"""
source_client = network.client_dir+"/source"
target_client = network.client_dir+"/target"

#source
## client creation and loading the model
fhemodel_src_client = FHEModelClient(source_client, key_dir=source_client)

## private and evaluation keys creation
fhemodel_src_client.generate_private_and_evaluation_keys()

## get the serialized evaluation keys
serialz_eval_keys_src = fhemodel_src_client.get_serialized_evaluation_keys()
print(f"Evaluation 'source' keys size: {len(serialz_eval_keys_src) / (10**6):.2f} MB")

## send public key to server
network.client_send_evaluation_key_to_server(serialz_eval_keys_src, "/source")
print("sourceeval_key_senttoserver")

#target
## client creation and loading the model
fhemodel_trgt_client = FHEModelClient(target_client, key_dir=target_client)

## private and evaluation keys creation
fhemodel_trgt_client.generate_private_and_evaluation_keys()

## get the serialized evaluation keys
serialz_eval_keys_trgt = fhemodel_src_client.get_serialized_evaluation_keys()
print(f"Evaluation 'target' keys size: {len(serialz_eval_keys_trgt) / (10**6):.2f} MB")

## send public key to server
network.client_send_evaluation_key_to_server(serialz_eval_keys_trgt, "/target")
print("target_eval_key_senttoserver")
```
<br>

### :o: How to deploy the FHE models.

 **In your remote terminal**:
- ```cd fhechess```
- the following folders must be empty. if not, emptied them!:
    - ```fhechess/deploy```
    - ```fhechess/client/source```
    - ```fhechess/client/target```
    - ```fhechess/server/model/client```
    - ```fhechess/server/model/target```
    <br> *(<ins>understanding</ins>: the ```deploy``` folder is like a basecamp to dispatch the model between ```client``` and ```server```. It can be deleted after deployment is completed.)*
- run ```$ python3 client_server_fhe_deploy.py```<br>

Toubleshooting: if a crash occurs because of an unemptied folder, the hard solution is to delete the said folder and re-create it with the same name.

While the deployment runs, you'll noticed some prints which keep you informed of the process. It takes fair minutes.

Here is an example of the contents of ```deploy, client, server``` folders when the deployment is over:<br>

**IMPORTANT: You will not have the same encryption folders 10868118508151244911 and 16067062207993741669. These folders will necessary be different for each machine and users.**

```python
fhechess
├── deploy
│   ├── source
│   │   ├── client.zip
│   │   ├── server.zip
│   │   └── versions.json
│   └── target
│       ├── client.zip
│       ├── server.zip
│       └── versions.json

├── client
│   ├
│   ├
│   ├── source
│   │   ├── 10868118508151244911
│   │   │   └── 0_0
│   │   │       ├── ksKey_0
│   │   │       ├── pbsKey_0
│   │   │       ├── pksKey_0
│   │   │       ├── secretKey_0
│   │   │       └── secretKey_1
│   │   └── client.zip
│   └── target
│       ├── 16067062207993741669
│       │   └── 0_0
│       │       ├── ksKey_0
│       │       ├── pbsKey_0
│       │       ├── pksKey_0
│       │       ├── secretKey_0
│       │       └── secretKey_1
│       └── client.zip

├── server
│   ├
│   ├
│   ├── model
│   │   ├── source
│   │   │   ├── serialized_evaluation_keys.ekl
│   │   │   └── server.zip
│   │   └── target
│   │       ├── serialized_evaluation_keys.ekl
│   │       └── server.zip
```