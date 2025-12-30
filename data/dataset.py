import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict
import albumentations as A


class MelanomaDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: Optional[A.Compose] = None) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = row['image_path']
        label = row['binary_label']
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }