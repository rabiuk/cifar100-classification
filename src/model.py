import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights


def get_model(num_classes=100):
    """
    Returns a ResNet18 model pre-configured for CIFAR-100 classification.
    """
    # Specify the weights
    weights = ResNet50_Weights.DEFAULT  # Use the default pre-trained weights

    # Load ResNet50 model with pre-trained weights
    model = resnet50(weights=weights)

    # Adjust the first convolution layer for CIFAR-100 if needed
    # Since CIFAR-100 images are 32x32, consider modifying the initial layers
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')  # Re-initialize

    model.maxpool = nn.Identity()  # Remove the maxpool layer

    # Adjust the final fully connected layer for 100 classes
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add a Dropout layer before the final linear layer
        nn.Linear(model.fc.in_features, num_classes)  # Adjust for the number of classes
    )

    return model

def load_pretrained_weights(model, weights):
    """
    Load pre-trained weights, excluding incompatible layers.
    """
    pre_trained_model = resnet50(weights=weights)
    pre_trained_dict = pre_trained_model.state_dict()
    model_dict = model.state_dict()

    # Exclude 'conv1.weight' since it's incompatible
    pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k != 'conv1.weight' and k in model_dict}

    # Update the model's state dict
    model_dict.update(pre_trained_dict)
    model.load_state_dict(model_dict)

    return model
