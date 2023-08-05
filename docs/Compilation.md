# Compilation & Simulation (Virtual Library)

At this step, if you need a deep dive into Compilation?! You can have look at [Zama's compilation explanations](https://docs.zama.ai/concrete-ml/advanced-topics/compilation).<br>

As we have made a custom quantized Brevitas models with Quantization Aware Training we use ```compile_brevitas_qat_model()``` to obtain "quantized_module" (a compiled version of our quantized model) for each of our models.

In this project, compilation can be found when:<br>

- **Testing**: in [launch_(test)_compile_fhe](../server_cloud/traintest_only/launch_(test)_compile_fhe.py) 

```python
#...
q_module_vl = compile_brevitas_qat_model(model, train_input, n_bits={"model_inputs":4, "model_outputs":4})
#...
```
where ```model``` is loaded quantized model (with dict items and weights), ```train_data``` is the transformed (see. "Special step: Compilation" in [Data Transformation](data_explanation.md)) "solo" input_data for Source model or multiple input_data for Target model.<br>
And finally ```n_bits``` are the maximum necessary bits used during training for input_data and weights.

Later on, the "quantized_module" is used in [test_model_FHE.py](../server_cloud/traintest_only/test_model_FHE.py).

```python
# concerning def test_source_concrete() module

for idx, (data, target) in loop_vlfhe_test:
      # from tensor to numpy
      data = data.cpu().detach().numpy()
      target = target.cpu().detach().numpy()
    
      # Quantize the inputs and cast to appropriate data type
      x_test_q = quantized_module.quantize_input(data).astype(dtype_inputs)

      # Accumulate the ground truth labels
      y_pred = quantized_module.quantized_forward(x_test_q, fhe="simulate")
      output = quantized_module.dequantize_output(y_pred)
```

- **Deploying** in 2 cases:

    * **simulate FHE circuit**
    concerns [compile_fhe_inprod.py](../server_cloud/server/compile_fhe_inprod.py).

    ```python
    # COMPILATION SECTION
        
    print("Concrete-ml is compiling")

    start_compile = time.time()

    # source model compilation
    self.compiled_source = compile_brevitas_qat_model(source_model, source_train_input, n_bits={"model_inputs":4, "model_outputs":4})
    end_compile_1 = time.time()
    print(f"Source_M compilation finished in {end_compile_1 - start_compile:.2f} seconds")

    # target model compilation
    self.compiled_target = compile_brevitas_qat_model(target_model, target_train_input, n_bits={"model_inputs":4, "model_outputs":4})
    end_compile_2 = time.time()
    print(f"Target_M compilation finished in {end_compile_2 - start_compile:.2f} seconds")

    #Checking that the network is compatible with FHE constraints
    print("checking FHE constraints compatibility")

    bitwidth_source = self.compiled_source.fhe_circuit.graph.maximum_integer_bit_width()
    bitwidth_target = self.compiled_target.fhe_circuit.graph.maximum_integer_bit_width()
    print(
    f"Max bit-width: source {bitwidth_source} bits, target {bitwidth_target} bits" + " -> Fine in FHE!!"
    if bitwidth_source <= 16 and bitwidth_target <= 16
    else f"{bitwidth_source} or {bitwidth_target} bits too high for FHE computation"
    )
    ```
    <br>

    Source model is 14 bits out of 16 (max bits) and Target model is 11. Then we have the confirmation that the models can be used in a FHE circuit.<br>
    
    **NB**: the compiled model (or quantized_module are loaded ONLY once when the Server is initialized. This is the when running [/server/server_all.py](../server_cloud/server/server_all.py) or [/server/server_simfhe.py](../server_cloud/server/server_simfhe.py).<br>

    * deploying FHE

    *   Run test (model are compiled with Concrete's FHE compiler to run inference on encrypted data): blob/quant_fhe/server_cloud/traintest_only/launch_(test)_compile_fhe.py)

    ## Model Deployment

    *   client generates private keys and a public evaluation key (used by the model's FHE evaluation on the server) and then encrypts data and decrypts results.

    *   server runs compiled model, makes inference on encrypted data.
