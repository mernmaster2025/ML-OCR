import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from model import OCRNet
from dataset import CLASSES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model once, reuse for every prediction
model = OCRNet(num_classes=62)
checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1736,), (0.3317,)),
])

def preprocess(image: Image.Image) -> torch.Tensor:
    """Normalize any input image to match EMNIST format."""
    image = image.convert('L')          # grayscale
    image = ImageOps.invert(image)      # EMNIST: white letter on black bg
    image = ImageOps.autocontrast(image)  # maximize contrast
    image = image.resize((28, 28), Image.LANCZOS)
    # torchvision serves EMNIST in its stored orientation, which is the
    # transpose of how the character reads. Match it — that is what the
    # model was trained on.
    image = image.transpose(Image.Transpose.TRANSPOSE)
    tensor = transforms.ToTensor()(image)
    tensor = transforms.Normalize((0.1736,), (0.3317,))(tensor)
    return tensor.unsqueeze(0)          # add batch dimension → (1,1,28,28)

def predict(image: Image.Image, top_k=5):
    """Return top-k predictions with confidence scores."""
    tensor = preprocess(image).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(top_k)

    results = [
        {
            'character': CLASSES[idx.item()],
            'confidence': f"{prob.item()*100:.1f}%",
            'prob': prob.item(),
        }
        for prob, idx in zip(top_probs, top_indices)
    ]
    return results