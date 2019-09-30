import os

import torch
from torch.utils.data import Dataset
from PIL import Image

class Data(Dataset):
    """Dataset class, used when inputting images from a directory"""
    def __init__(self, data_dir, transform=None):
        photos = os.listdir(data_dir)
        self.data_dir = data_dir
        self.images = [photo for photo in photos]
        self.transform = transform
        
    def __getitem__(self, index):
        photo = os.path.join(self.data_dir, self.images[index])
        img = Image.open(photo)
        img = self.transform(img)
        return img, self.images[index]
    
    def __len__(self):
        return len(self.images)
    