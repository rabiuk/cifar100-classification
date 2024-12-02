import torch
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from torchvision.transforms import RandAugment

# # CIFAR-100 normalization parameters
# CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
# CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# ImageNet normalization parameters
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_transforms(data_augmentation=False):
    if data_augmentation:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            RandAugment(),  # Use RandAugment for more diverse augmentation
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transform


def load_data(batch_size, data_augmentation=False):
    """Load CIFAR-100 dataset and return dataloaders."""
    train_transform = get_transforms(data_augmentation)
    test_transform = get_transforms(data_augmentation=False)
    
    # Datasets
    train_dataset = datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root='../data', train=False, download=True, transform=test_transform)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def preprocess_test_data(test_csv_path):
    """
    Load and preprocess the custom test set from test.csv.
    """
    # Load the test.csv
    test_df = pd.read_csv(test_csv_path)

    # Extract pixel values (excluding the ID column)
    pixel_data = test_df.iloc[:, 1:].values.astype(np.float32)

    # Reshape to (num_samples, 3, 32, 32)
    images = pixel_data.reshape(-1, 3, 32, 32)

    # Scale pixel values to [0, 1]
    images /= 255.0

    # Normalize using ImageNet mean and std
    images = (images - np.array(IMAGENET_MEAN).reshape(3, 1, 1)) / np.array(IMAGENET_STD).reshape(3, 1, 1)

    # Convert to PyTorch tensor
    images_tensor = torch.from_numpy(images)

    return test_df['ID'], images_tensor
