import torch
import pandas as pd
from src.data_loader import preprocess_test_data

def predict_and_save(model, test_csv_path, submission_path):
    """
    Predict labels for the custom test set and save the results in submission.csv.
    """
    # Preprocess the test data
    ids, test_images = preprocess_test_data(test_csv_path)

    # Convert test images to the same dtype as the model's weights
    test_images = test_images.float()

    # Ensure the model is in evaluation mode
    model.eval()

    # Run predictions
    with torch.no_grad():
        predictions = model(test_images).argmax(dim=1).numpy()  # Get predicted class labels

    # Create the submission DataFrame
    submission_df = pd.DataFrame({
        'ID': ids,
        'LABEL': predictions
    })

    # Save to CSV
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

# Example usage (replace 'model' with your trained model object)
# predict_and_save(model, '../data/test.csv', '../outputs/predictions/submission.csv')
