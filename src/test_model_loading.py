import os
import torch
from src.model import get_model
from src.utils import load_config


def test_model_loading(model_save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=100)
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "configs", "default.yaml")
    config = load_config(config_path)

    print(f"Loaded Configurations: {config}")
    # Update the path to your saved model
    model_save_path = config['paths']['model_save_path']
    test_model_loading(model_save_path)
