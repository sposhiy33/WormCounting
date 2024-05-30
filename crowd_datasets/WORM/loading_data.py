import torch
import torchvision.transforms as standard_transforms
from .WORM import WORM

def loading_data(data_root):

    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])
  
    # "patch" must be set to true in order to ensure that images fit the correct aspect ratio
    train_set = WORM(data_root, train=True, transform=transform, patch=True, rotate=True, scale=False, flip=False)
    val_set = WORM(data_root, train=False, transform=transform)

    return train_set, val_set
