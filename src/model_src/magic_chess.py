from dataset import ZDataset

from cnn_model import Net

model = Net()

if train_on_gpu:
    model.cuda()