import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_loaders, CLASSES
from model import OCRNet

def train(epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    model     = OCRNet(num_classes=62).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Reduce learning rate when accuracy stops improving
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        factor=0.5, patience=3
    )

    train_loader, test_loader = get_loaders()

    best_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss, correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_loader.dataset) * 100

        # Evaluation
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_correct += (outputs.argmax(1) == labels).sum().item()

        test_acc = test_correct / len(test_loader.dataset) * 100

        # Update learning rate based on test accuracy
        scheduler.step(test_acc)

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Test: {test_acc:.1f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'accuracy': best_acc,
            }, 'checkpoints/best_model.pth')
            print(f"  → New best saved: {best_acc:.2f}%")

    print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")