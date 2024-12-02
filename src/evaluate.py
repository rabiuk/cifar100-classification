import os
import torch
import pandas as pd
from src.model import get_model
from src.data_loader import preprocess_test_data
from src.utils import load_config

def load_model(config, device):
    """
    Load the trained model and prepare it for evaluation.
    """
    model_save_path = config['paths']['model_save_path']
    model = get_model(num_classes=100)
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))

    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_save_path}")
    return model

def generate_submission(config, model, device):
    """
    Generate predictions for the custom test set and save them to a submission file.
    """
    # Path to test.csv
    test_csv_path = config['paths']['test_csv_path']
    submission_path = config['paths']['submission_path']

    # Preprocess the test data
    ids, test_images = preprocess_test_data(test_csv_path)
    test_images = test_images.to(device)

    # Predict class labels
    with torch.no_grad():
        predictions = model(test_images).argmax(dim=1).cpu().numpy()  # Convert to numpy

    # Create the submission DataFrame
    submission_df = pd.DataFrame({
        'ID': ids,
        'LABEL': predictions
    })

    # Save to CSV
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    # Load configurations
    # Dynamically construct the path to the configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
    config_path = os.path.join(script_dir, "..", "configs", "default.yaml")  # Relative to script_dir
    config = load_config(config_path)

    print(f"Loaded Configurations: {config}")

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = load_model(config, device)

    # Generate submission file
    generate_submission(config, model, device)
