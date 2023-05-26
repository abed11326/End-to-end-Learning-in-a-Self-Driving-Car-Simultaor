import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import hflip
import numpy as np
import cv2
from hypParam import device, val_percent

array2tensor = ToTensor()

def train_val_split(data_log_path):
    data_csv = np.loadtxt(data_log_path, delimiter=",", dtype=str)[:, :4]
    train, val = random_split(data_csv, [1 - val_percent, val_percent])
    return train, val

def process_image(im):
    im = cv2.resize(im, (200, 66), interpolation = cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    return array2tensor(im)

def get_image(path):
    im = cv2.imread(path)
    return process_image(im)


class Data(Dataset):
    def __init__(self, data_log, training):
        super(Data, self).__init__()
        self.data_log = data_log
        self.training = training
        self.load_data()
        
        if not training:
            self.data = torch.stack(self.data).to(device)
            self.labels = torch.stack(self.labels).to(device)

    def load_data(self):
        self.data = []
        self.labels = []
        for item in self.data_log:
            label = torch.tensor(float(item[3]), dtype=torch.float32)
            for i in range(3):
                self.data.append(get_image(item[i]))
            self.labels.append(label)
            self.labels.append(label + 0.15)
            self.labels.append(label - 0.15)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        if self.training:
            if np.random.rand() < 0.5:
                image = hflip(image)
                label *= -1
        return image, label