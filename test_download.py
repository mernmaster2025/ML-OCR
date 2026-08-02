from torchvision import datasets, transforms
import os

print("Starting download...", flush=True)

os.makedirs('./data', exist_ok=True)

transform = transforms.ToTensor()

print("Downloading EMNIST...", flush=True)

try:
    data = datasets.EMNIST(
        './data',
        split='byclass',
        train=True,
        download=True,
        transform=transform
    )
    print(f"Success! {len(data)} samples loaded.", flush=True)

except Exception as e:
    print(f"Error: {e}", flush=True)