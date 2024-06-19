"""
load torch Dataset:
    ER: Endoplasmic Reticulum
    MITO: Mitochondria
"""

import cv2
import torch
import numpy as np
import albumentations as albu
from torch.utils.data import Dataset

from config import ER_train_path_list, ER_val_path_list, ER_test_path_list
from config import MITO_train_path_list, MITO_val_path_list, MITO_test_path_list

class Dataset(Dataset):
    def __init__(self, path_list : list[dict[str, dict[str, str]]], transform : bool = False) -> None:
        """
        Custom dataset class for loading image and mask pairs.

        Args:
            path_list: List of dictionaries containing image and mask paths.
            transform: Optional transform to be applied to the data.
        """
        super().__init__()
        self.path_list = path_list
        self.transform = transform
    
    def __min_max_normalize__(self, data : np.ndarray) -> np.ndarray:
        """
        Normalize the data to the range of [0, 1] using min-max normalization.
        """
        data_min = np.min(data)
        data_max = np.max(data)
        data_norm = (data - data_min) / (data_max - data_min) if data_max !=  data_min else np.ones_like(data)
        return data_norm

    def __getitem__(self, idx : int) -> tuple[str, torch.Tensor, torch.Tensor]:
        """
        Retrieves the image and mask at the specified index.

        Args:
            idx: Index of the data sample to retrieve.

        Returns:
            A tuple containing the name, image tensor, and mask tensor.
        """
        name_image_mask_pair = self.path_list[idx]
        name = next(iter(name_image_mask_pair.keys()))
        image_path = name_image_mask_pair[name]['image']
        mask_path = name_image_mask_pair[name]['mask']
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # uinit16 [0-65535]
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)   # unit8   [0-255]
        image = self.__min_max_normalize__(image).astype(np.float32) # float32
        mask[mask > 0] = 1 # binary mask
        assert image.shape == mask.shape, 'Image and mask should have the same shape'
        if self.transform:
            train_transform = albu.Compose([
                albu.RandomRotate90(),
                albu.Flip(),
                albu.Resize(image.shape[0], image.shape[1]),
            ])
            augmented = train_transform(image=image, mask=mask)
            image = augmented['image']
            mask  = augmented['mask']
        image = torch.from_numpy(image).unsqueeze(0)
        mask  = torch.from_numpy(mask).unsqueeze(0)
        return name, image, mask
    
    def __len__(self) -> int:
        """
        Returns the total number of image and mask pairs in the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.path_list)

ER_train_dataset = Dataset(path_list=ER_train_path_list, transform=True)
ER_val_dataset   = Dataset(path_list=ER_val_path_list  , transform=False)
ER_test_dataset  = Dataset(path_list=ER_test_path_list , transform=False)

MITO_train_dataset = Dataset(path_list=MITO_train_path_list, transform=True)
MITO_val_dataset   = Dataset(path_list=MITO_val_path_list  , transform=False)
MITO_test_dataset  = Dataset(path_list=MITO_test_path_list , transform=False)
