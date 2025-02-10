import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class DataHandler:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def get_nyud_dataset(self):
        # Assuming NYUD dataset is downloaded and organized in data_dir
        train_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'train'),
            transform=self.transform
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'val'),
            transform=self.transform
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader