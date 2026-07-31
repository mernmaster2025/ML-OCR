import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(batch_size=128): # 128 images at once per training step
    # Augmentation only on training data
    train_transform = transforms.Compose([
        transforms.ToTensor(), # Converts a PIL image (pixel values 0–255) into a PyTorch tensor (values 0.0–1.0)
        transforms.Normalize((0.1736,), (0.3317,)),   # EMNIST mean/std
        transforms.RandomRotation(10),                 # slight rotation
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),                      # small shifts
            shear=5                                    # slight lean
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3317,)),
    ])

    # 'byclass' split = 62 classes (0-9, A-Z, a-z)
    train_data = datasets.EMNIST(
        './data', split='byclass',
        train=True, download=True,
        transform=train_transform
    )
    test_data = datasets.EMNIST(
        './data', split='byclass',
        train=False, download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size,
        shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size,
        shuffle=False, num_workers=2
    )

    return train_loader, test_loader


# Class label mapping: 0-9 → digits, 10-35 → A-Z, 36-61 → a-z
CLASSES = (
    [str(i) for i in range(10)] +
    [chr(i) for i in range(65, 91)] +   # A–Z
    [chr(i) for i in range(97, 123)]    # a–z
)