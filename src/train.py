import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import get_model, load_pretrained_weights
from src.data_loader import load_data
from src.utils import load_config
import os
from torchvision.models import ResNet50_Weights

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Processing batch {batch_idx + 1}/{len(dataloader)}", end='\r', flush=True)  # Debug

        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track accuracy and loss
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100.0 * correct / total
    return epoch_loss, epoch_accuracy

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track accuracy and loss
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(dataloader)
    val_accuracy = 100.0 * correct / total
    return val_loss, val_accuracy

def train_model(config):
    """
    Main training function to train and validate the model.
    """
    print("Starting train_model function...")  # Debug print
    
    # Unpack training configurations
    num_epochs = config['train']['num_epochs']
    batch_size = config['train']['batch_size']
    learning_rate = config['train']['learning_rate']
    weight_decay = config['train']['weight_decay']
    momentum = config['train']['momentum']
    data_augmentation = config['train']['data_augmentation']
    
    # Resolve project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    # Construct model_save_path relative to project_root
    model_save_path = os.path.join(project_root, config['paths']['model_save_path'])
    model_save_path = os.path.abspath(model_save_path)  # Get absolute path
    
    # Ensure the directory exists
    model_save_dir = os.path.dirname(model_save_path)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # # Debug prints
    # print(f"script_dir: {script_dir}")
    # print(f"project_root: {project_root}")
    # print(f"model_save_path: {model_save_path}")
    print(f"Model will be saved to: {model_save_path}")

    # # Test saving functionality
    # test_save_path = os.path.join(model_save_dir, "test_file.txt")
    # with open(test_save_path, 'w') as f:
    #     f.write("This is a test file.")
    # print(f"Test file saved to {test_save_path}")
    
    print("Loading data...")  # Debug print
    train_loader, val_loader = load_data(batch_size=batch_size, data_augmentation=data_augmentation)
    print("Data loaded successfully")  # Debug print

    print("Initializing model...")  # Debug print
    model = get_model(num_classes=100)
    model = load_pretrained_weights(model, ResNet50_Weights.DEFAULT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")  # Debug print
    model = model.to(device)
    print("Model initialized and moved to device")  # Debug print

    # **Unfreeze Earlier Layers**
    for param in model.parameters():
        param.requires_grad = True

    print("All layers of the model are unfrozen and will be trained.")  # Debug print

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler =  optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=num_epochs)

    
    print("Optimizer and scheduler initialized")  # Debug print

    print("Starting training loop...")  # Debug print

    # Early stopping parameters
    patience = 20  # Number of epochs to wait after last improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on validation set
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save the best model
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch {epoch+1}: Improvement in validation loss. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}: No improvement in validation loss for {epochs_no_improve} epoch(s).")

        # Early stopping condition
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    # Update the path to be relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "configs", "default.yaml")
    config = load_config(config_path)
    model_save_path = os.path.join(script_dir, "..", config['paths']['model_save_path'])

    print(f"Loaded Configurations: {config}")
    
    # Ensure output directories exist
    output_dir = os.path.join(script_dir, "..", "outputs", "models")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model will be saved to: {model_save_path}")  # Debug print

    shutil.copy(config_path, os.path.join(output_dir, "config_used.yaml"))

    train_model(config=config)