import torch

fm = [24, 36, 48, 64, 64] # CNN Feature maps
fc = [100, 50, 10] # Fully-connected layers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
no_workers = 8
no_epochs = 20
lr = 0.0001
val_percent = 0.1
