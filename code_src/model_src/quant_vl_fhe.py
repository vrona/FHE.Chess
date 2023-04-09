import numpy as np
from concrete.numpy.compilation.configuration import Configuration
from concrete.ml.torch.compile import compile_brevitas_qat_model
from concrete.ml.quantization import quantized_module
"""
input: float --> quantization --> encryption
output: float <-- quantization <-- decryption
"""

# QUANTIZATION
clear_data = float(0)

# quantization
quantized_data = quantized_module.QuantizedModule.quantize_input(clear_data)

for i in range(quantized_data.shape[0]):
    # batch is 1 dim
    q = np.expand_dims(quantized_data[i,:],0)

    # execute model in FHE
    out_fhe = quantized_module.QuantizedModule.forward_fhe(q)

    # de-quantization
    out_clear = quantized_module.QuantizedModule.dequantize_output(out_fhe)


# SIMULATION
# enabling virtual library via compilation configuration

comp_config_vl = Configuration(
    dump_artifacts_on_unexpected_failures=False,
    enable_unsafe_features=True,
)

compile_brevitas_qat_model(some_net, some_data, some_n_bits, use_virtual_lib=True,
configuration=comp_config_vl)