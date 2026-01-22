import torch
from torch import nn, optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

if __name__ == '__main__':
    # Where everything lives
    DATASET_PATH = "/Users/kushdesai/waste-garbage-management-dataset"
    SAVE_PATH = "/Users/kushdesai/Desktop/best_material_classifier.pth"
    CHECKPOINT_DIR = "/Users/kushdesai/Desktop/checkpoints"

    # Create checkpoint directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Check if we have a GPU available (makes training way faster)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # The main material categories we care about
    MATERIALS = ["plastic", "paper", "metal", "glass", "organic", "other"]

    # Maps the specific trash types in our dataset to broader material categories
    # For example, both "paper" and "cardboard" get grouped as "paper"
    CLASS_TO_MATERIAL = {
        "metal": "metal",
        "glass": "glass",
        "biological": "organic",
        "paper": "paper",
        "cardboard": "paper",
        "plastic": "plastic",
        "battery": "other",
        "trash": "other",
        "shoes": "other",
        "clothes": "other"
    }

    # Training transform - we want to augment the data to make the model more robust
    # Things like flipping and rotating help the model learn to recognize trash from different angles
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),  # sometimes flip the image
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # vary lighting conditions
        transforms.RandomRotation(15),  # rotate a bit - trash isn't always perfectly aligned
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])  # ImageNet stats
    ])

    # Validation transform - no augmentation here, we want to test on "normal" images
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Load the dataset with training transforms first
    full_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)

    # Figure out how to map from the dataset's class indices to our material categories
    idx_to_material = {i: CLASS_TO_MATERIAL[c] for c, i in full_dataset.class_to_idx.items()}

    # Custom dataset wrapper that converts trash type labels to material labels
    class MaterialDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset):
            self.base = base_dataset
        
        def __len__(self):
            return len(self.base)
        
        def __getitem__(self, idx):
            img, cls_idx = self.base[idx]
            # Convert the specific trash type to a material category
            material = idx_to_material[cls_idx]
            label_idx = MATERIALS.index(material)
            return img, label_idx

    dataset = MaterialDataset(full_dataset)

    # Split into training (80%) and validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Now apply the correct transforms to each split
    train_dataset.dataset.base.transform = train_transform
    val_dataset.dataset.base.transform = val_transform

    # Create data loaders - these handle batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Dataset loaded: {train_size} training images, {val_size} validation images")

    # Load pre-trained MobileNetV2 - it's already learned to recognize tons of visual features
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all the pre-trained layers - we don't want to mess with what it already knows
    for param in model.parameters():
        param.requires_grad = False

    # Remove the original classifier and use the model as a feature extractor
    model.classifier = nn.Identity()
    model.to(device)
    model.eval()  # keep it in eval mode since we're not training it

    # Our custom classifier that learns to identify materials from the features
    # Added dropout to prevent overfitting - basically randomly turns off some neurons during training
    material_classifier = nn.Sequential(
        nn.Linear(1280, 128),  # MobileNetV2 outputs 1280 features
        nn.ReLU(),
        nn.Dropout(0.5),  # helps the model generalize better
        nn.Linear(128, len(MATERIALS))  # output layer - one score per material
    )
    material_classifier.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(material_classifier.parameters(), lr=1e-3)

    # Learning rate scheduler - reduces learning rate if validation accuracy plateaus
    # This helps fine-tune the model after initial learning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    # Training setup
    EPOCHS = 15  # increased from 5 - gives the model more time to learn
    best_val_acc = 0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 5  # stop if no improvement for 5 epochs

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        # === TRAINING PHASE ===
        material_classifier.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Extract features using the frozen MobileNet
            with torch.no_grad():
                features = model(imgs)
            
            # Pass features through our classifier
            outputs = material_classifier(features)
            loss = criterion(outputs, labels)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total * 100
        train_loss = running_loss / total
        
        # === VALIDATION PHASE ===
        material_classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                
                # Extract features and classify
                features = model(imgs)
                outputs = material_classifier(features)
                loss = criterion(outputs, labels)
                
                # Track metrics
                val_loss += loss.item() * imgs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                # Save predictions for detailed metrics later
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = val_correct / val_total * 100
        val_loss = val_loss / val_total
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.1f}%")
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_acc)
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(material_classifier.state_dict(), SAVE_PATH)
            print("  ✓ Best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save a checkpoint every 3 epochs (just in case something crashes)
        if (epoch + 1) % 3 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': material_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved (epoch {epoch+1})")
        
        # Early stopping - quit if we're not improving
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nNo improvement for {EARLY_STOP_PATIENCE} epochs. Stopping early.")
            break
        
        print()

    print("="*60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.1f}%")
    print("="*60)

    # Save a final checkpoint with all the training info
    final_checkpoint = os.path.join(CHECKPOINT_DIR, "final_checkpoint.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': material_classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'final_val_acc': val_acc,
    }, final_checkpoint)
    print(f"\n💾 Final checkpoint saved to: {final_checkpoint}")


    # === DETAILED PERFORMANCE ANALYSIS ===
    print("\nDetailed Performance Metrics:")
    print("-" * 60)

    # Classification report shows precision, recall, and F1-score for each material
    print("\nPer-Material Performance:")
    print(classification_report(all_labels, all_preds, target_names=MATERIALS, digits=3))

    # Confusion matrix shows which materials get confused with each other
    print("\nConfusion Matrix:")
    print("(Rows = actual material, Columns = predicted material)")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # Show which materials are hardest to classify
    print("\nPer-Material Accuracy:")
    for i, material in enumerate(MATERIALS):
        if cm.sum(axis=1)[i] > 0:  # avoid division by zero
            acc = cm[i, i] / cm.sum(axis=1)[i] * 100
            print(f"  {material:10s}: {acc:5.1f}%")

    print("\n" + "="*60)
    print("Model saved to:", SAVE_PATH)
    print("Ready for testing! Good luck with your science fair project!")
    print("="*60)
