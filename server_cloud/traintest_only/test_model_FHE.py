import torch
import numpy as np
import wandb
from tqdm import tqdm

#torch.manual_seed(42)

# CUDA's availability

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


wandb.init(
        project = "Chess_App",

        config = {
        "learning_rate": 1.0e-3,
        "architecture": "CNN",
        "dataset": "White Black ELO 2000 A.Revel kaggle dataset",
        "epochs": 5,
        }
    )

# copy config
wb_config = wandb.config


# 🅥🅘🅡🅣🅤🅐🅛 🅕🅗🅔

## source
def test_source_concrete(quantized_module, test_input,dtype_inputs=np.int64):
    """Test a neural network that is quantized and compiled with Concrete-ML."""

  
    print("beginning of test")
    # Casting the inputs into int64 is recommended
    # all_data = np.zeros((len(test_input)), dtype=dtype_inputs)
    # all_targets = np.zeros((len(test_input)), dtype=dtype_inputs)
    loop_vlfhe_test = tqdm(enumerate(test_input), total=len(test_input), leave=False)
    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector

    accuracy = 0
    #for data, target in zip(test_input, test_target):
    for idx, (data, target) in loop_vlfhe_test:
      # from tensor to numpy
      data = data.cpu().detach().numpy()
      target = target.cpu().detach().numpy()
    
      # Quantize the inputs and cast to appropriate data type
      x_test_q = quantized_module.quantize_input(data).astype(dtype_inputs)

      # Accumulate the ground truth labels
      y_pred = quantized_module.quantized_forward(x_test_q, fhe="simulate")
      output = quantized_module.dequantize_output(y_pred)

      # Take the predicted class from the outputs and store it
      y_pred = np.argmax(output, 1)
      y_targ = np.argmax(target, 1)

      accuracy += np.sum(y_pred == y_targ)

      wandb.log({"accuracy": 100 * accuracy / len(test_input)})
      loop_vlfhe_test.set_description(f"test [{idx}/{len(test_input)}]")
      loop_vlfhe_test.set_postfix(acc = accuracy)#acc_rate = 100 * accuracy / len(testloader))

    # closing the wandb logs
    wandb.finish()


## target
def test_target_concrete(quantized_module, test_input,dtype_inputs=np.int64):
    """Test a neural network that is quantized and compiled with Concrete-ML."""

  
    print("beginning of test")
    # Casting the inputs into int64 is recommended
    # all_data = np.zeros((len(test_input)), dtype=dtype_inputs)
    # all_targets = np.zeros((len(test_input)), dtype=dtype_inputs)
    loop_vlfhe_test = tqdm(enumerate(test_input), total=len(test_input), leave=False)
    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector

    accuracy = 0
    #for data, target in zip(test_input, test_target):
    for idx, (chessboard, source, target) in loop_vlfhe_test:
      # from tensor to numpy
      chessboard = chessboard.cpu().detach().numpy()
      source = source.cpu().detach().numpy()
      target = target.cpu().detach().numpy()
      

      # Quantize the inputs and cast to appropriate data type
      chessb_q, source_q = quantized_module.quantize_input(chessboard, source)#.astype(dtype_inputs)
      #source_q = quantized_module.quantize_input(source).astype(dtype_inputs)

      # Accumulate the ground truth labels
      y_pred = quantized_module.quantized_forward(chessb_q, source_q, fhe="simulate")
      output = quantized_module.dequantize_output(y_pred)


      # Take the predicted class from the outputs and store it
      y_pred = np.argmax(output, 1)
      y_targ = np.argmax(target, 1)

      accuracy += np.sum(y_pred == y_targ)

      wandb.log({"accuracy": 100 * accuracy / len(test_input)})
      loop_vlfhe_test.set_description(f"test [{idx}/{len(test_input)}]")
      loop_vlfhe_test.set_postfix(acc = accuracy, targ=y_targ, pred=y_pred)#acc_rate = 100 * accuracy / len(testloader))

    # closing the wandb logs
    wandb.finish()