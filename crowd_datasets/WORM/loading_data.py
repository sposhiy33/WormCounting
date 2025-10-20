import torch
import torchvision.transforms as standard_transforms
import torchvision.transforms.functional as F

from .WORM import WORM, WORM_eval


def loading_data(
    data_root,
    num_patch,
    patch_size,
    multiclass=False,
    equal_crop=False,
    scale=False,
    hsv=False,
    hse=False,
    edges=False,
    sharpness=False,
    patch=False,
    equalize=False,
    salt_and_pepper=False,
):
    
    # the pre-proccssing transform
    transform = standard_transforms.Compose(
        [
            standard_transforms.ColorJitter(
                brightness=[0,2], contrast=[0,2], saturation=[0,2], hue=[-0.5, 0.5]),         
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


    # "patch" must be set to true in order to ensure that images fit the correct aspect ratio
    train_set = WORM(
        data_root,
        train=True,
        transform=transform,
        patch=patch,
        num_patch=num_patch,
        patch_size=patch_size,
        rotate=False,
        scale=scale,
        flip=False,
        multiclass=multiclass,
        hsv=hsv,
        hse=hse,
        edges=edges,
        sharpness=sharpness,
        equalize=equalize,
        salt_and_pepper=salt_and_pepper,
    )
    val_set = WORM(
        data_root,
        train=False,
        transform=transform,
        num_patch=num_patch,
        patch_size=patch_size,
        multiclass=multiclass,
        hsv=hsv,
        hse=hse,
        edges=edges,
        sharpness=sharpness,
        equalize=equalize,
    )
    return train_set, val_set


def loading_data_val(
    data_root,
    num_patch=0,
    patch_size=0,
    multiclass=False,
    equal_crop=False,
    hsv=False,
    hse=False,
    edges=False,
    equalize=False,
):

    # the pre-proccssing transform
    transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    train_set = WORM(
        data_root,
        train=True,
        transform=transform,
        num_patch=num_patch,
        patch_size=patch_size,
        multiclass=multiclass,
        equal_crop=equal_crop,
        hsv=hsv,
        hse=hse,
        edges=edges,
        equalize=equalize,
    )

    val_set = WORM(
        data_root,
        train=False,
        transform=transform,
        num_patch=num_patch,
        patch_size=patch_size,
        multiclass=multiclass,
        equal_crop=equal_crop,
        hsv=hsv,
        hse=hse,
        edges=edges,
        equalize=equalize,
    )

    return train_set, val_set

def loading_data_eval(data_root):
    # the pre-proccssing transform
    transform = standard_transforms.Compose(
        [
            standard_transforms.Resize(1000),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    val_set = WORM_eval(
        data_root,
        transform=transform,
    )

    return val_set

def load_viz_data(data_root, multiclass=False, hsv=False, hse=False, edges=True):

    # transform to tensor
    transform = standard_transforms.ToTensor()

    train_set = WORM(
        data_root,
        train=True,
        transform=transform,
        patch=True,
        rotate=False,
        scale=False,
        flip=False,
        multiclass=multiclass,
        hsv=hsv,
        hse=hse,
        edges=edges,
    )
    val_set = WORM(
        data_root,
        train=False,
        transform=transform,
        multiclass=multiclass,
        hsv=hsv,
        hse=hse,
        edges=edges,
    )

    return train_set, val_set
