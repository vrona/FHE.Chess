from dataset import ZDataset

model = Net()

if train_on_gpu:
    model.cuda()