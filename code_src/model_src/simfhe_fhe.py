import tqdm
import wandb
from concrete.numpy.compilation.configuration import Configuration
from concrete.ml.torch.compile import compile_brevitas_qat_model
from concrete.ml.quantization import quantized_module
"""
input: float --> quantization --> encryption
output: float <-- dequantization <-- decryption
"""

# CUDA's availability

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# wandb.init(
#         project = "Chess_App",

#         config = {
#         "learning_rate": 1.0e-3,
#         "architecture": "CNN",
#         "dataset": "White Black ELO 2000 A.Revel kaggle dataset",
#         "epochs": 5,
#         }
#     )

# copy config
#wb_config = wandb.config

cfg = Configuration(
        dump_artifacts_on_unexpected_failures=False,
        enable_unsafe_features=True,
        p_error=None,
        global_p_error=None)

q_module_vl = compile_brevitas_qat_model(model, trainloader, cfg, n_bits={"a_bits": 8, "w_bits":8},use_virtual_lib=True,configuration=cfg)

"""def test_concrete(model, testloader, use_fhe, use_vl):

    cfg = Configuration(
        dump_artifacts_on_unexpected_failures=False,
        enable_unsafe_features=True)
    

    test_loss = 0.0
    accuracy = 0
    
    
    loop_test = tqdm(enumerate(testloader), total=len(testloader), leave=False)
    
    model.eval().to(device)
    
    with torch.no_grad():
        
        for batch_idx, (data, target) in loop_test:

            data, target = data.to(torch.float32).to(device), target.to(torch.float32).to(device)
            # forward

            output = model(data)
            
            # batch loss
            loss = criterion(output, target)

            # test_loss update
            test_loss += loss.item()

            # accuracy (output vs target)
            outdix = output.argmax(1)
            tardix = target.argmax(1)


            accuracy += (outdix == tardix).sum().item()


            wandb.log({"test_loss": loss.item()})#, "accuracy": 100 * accuracy / len(testloader)})
            loop_test.set_description(f"test [{batch_idx}/{len(testloader)}]")
            loop_test.set_postfix(testing_loss = loss.item(), acc = accuracy)#)_rate = 100 * accuracy / len(testloader))


        # average test loss
        test_loss = test_loss/len(testloader)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        print('\nTest Accuracy: %2d%% ' % (100 * accuracy / len(testloader)))
    """
    # closing the wandb logs
    # wandb.finish()