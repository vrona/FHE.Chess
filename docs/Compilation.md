# Compilation & Simulation (Virtual Library)

At this step, if you need a deep dive into Compilation?! You can have look at [Zama's compilation explanations](https://docs.zama.ai/concrete-ml/advanced-topics/compilation).<br>

As we have made a custom quantized Brevitas models with Quantization Aware Training we use ```compile_brevitas_qat_model()``` to obtain our "quantized_module" (aka compiled (quantized) model)

In this project, compilation can be found when:<br>

- **Testing**: in [launch_(test)_compile_fhe](../server_cloud/traintest_only/launch_(test)_compile_fhe.py) 

```python
...
q_module_vl = compile_brevitas_qat_model(model, train_input, n_bits={"model_inputs":4, "model_outputs":4})
...
```

and then [test_model_FHE.py](../server_cloud/traintest_only/test_model_FHE.py) which is being called by .

test_source_concrete
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

- "deployment" in 2 cases: simulate FHE circuit and deploying FHE.

*   Run test (model are compiled with Concrete's FHE compiler to run inference on encrypted data): blob/quant_fhe/server_cloud/traintest_only/launch_(test)_compile_fhe.py)

## Model Deployment

*   client generates private keys and a public evaluation key (used by the model's FHE evaluation on the server) and then encrypts data and decrypts results.

*   server runs compiled model, makes inference on encrypted data.
