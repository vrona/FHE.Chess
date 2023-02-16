from dataset import ZDataset
from trainnvalid import train_valid, test
from cnn_model import Net

model = Net()


Dataset = "path/chess-game" # ZDataset(we_2000['AN'])
training_set = Dataset + "/train"
valid_set = Dataset + "/valid"
test_set = Dataset + "/test"


"""
LOADING SECTION
training_set = ZDataset(dataset['AN'])
"""


# normalization + convert to tensor
trainloader = torch.utils.data.DataLoader(training_set, batch_size=32, drop_last=True)
validloader = torch.utils.data.DataLoader(valid_set, batch_size=32, drop_last=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=32, drop_last=True)
