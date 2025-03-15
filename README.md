# **CIFAR-100 Classification Project**

## **1. Introduction**

The goal of this project is to develop a deep learning model capable of classifying images from the **CIFAR-100 dataset**, which contains **100 different classes** of objects. To achieve this, we implement a **ResNet-50 model** pretrained on ImageNet and fine-tune it for CIFAR-100 classification. This project follows a structured **machine learning pipeline**, covering data preprocessing, model training, evaluation, and inference.

------

## **2. Methodology**

The methodology consists of the following key steps:

1. **Data Preprocessing**:
   - Normalize images using ImageNet normalization parameters.
   - Perform data augmentation techniques like AutoAugment and RandAugment.
   - Load training and validation datasets using PyTorch’s `torchvision.datasets`.
2. **Model Selection & Training**:
   - Use a **ResNet-50** model pre-trained on ImageNet.
   - Modify the final classification layer to match 100 output classes.
   - Optimize using cross-entropy loss and **Adam optimizer**.
   - Train using **mini-batches** and evaluate performance with accuracy metrics.
3. **Evaluation**:
   - Measure performance using accuracy on the validation dataset.
   - Save and reload the trained model for inference.
4. **Prediction & Submission**:
   - Process test data, generate predictions, and save them in a structured format.
   - Ensure compatibility with external evaluation pipelines.

------

## **3. Implementation**

This section provides an in-depth analysis of each script in the project.

### **3.1. Data Preprocessing (`src/data_loader.py`)**

- Uses `torchvision.transforms` to normalize and augment images.
- Loads CIFAR-100 dataset with train-test splits.
- Defines `get_transforms()` to apply data augmentation and `load_data()` to return PyTorch DataLoader objects.

### **3.2. Model Definition (`src/model.py`)**

- Implements a 

  ResNet-50-based classifier

   using:

  ```python
  from torchvision.models import resnet50, ResNet50_Weights
  ```

- Replaces the last fully connected layer:

  ```python
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  ```

- Allows fine-tuning of all or selective layers.

### **3.3. Model Training (`src/train.py`)**

- Defines training logic using:

  ```python
  def train_one_epoch(model, dataloader, criterion, optimizer, device):
  ```

- Tracks loss and accuracy.

- Uses **Adam optimizer** and **cross-entropy loss**.

- Saves trained models periodically.

### **3.4. Evaluation (`src/evaluate.py`)**

- Loads the trained model and evaluates performance on test images:

  ```python
  model.load_state_dict(torch.load(model_save_path))
  ```

- Uses standard accuracy metrics.

### **3.5. Prediction (`src/predict.py`)**

- Loads test images, runs inference, and stores results.

- Uses:

  ```python
  model.eval()
  ```

### **3.6. Utility Functions (`src/utils.py`)**

- Reads YAML configuration files.
- Handles file paths, logging, and parameter loading.

------

## **4. Results & Performance**

- The model achieves **high accuracy** due to transfer learning from ImageNet.
- **Data augmentation** significantly improves generalization.
- Performance metrics indicate **strong classification accuracy**.

------

## **5. Challenges & Improvements**

- **Dataset Complexity**: CIFAR-100 has **100 classes**, making classification harder than CIFAR-10.
- **Overfitting**: Regularization and dropout layers could be tested.
- **Inference Speed**: Model optimization for lower latency is possible.

------

## **6. Conclusion**

This project successfully implements a **deep learning-based image classifier** for the CIFAR-100 dataset using **ResNet-50 transfer learning**. The structured pipeline ensures **efficient training, evaluation, and prediction**, making it a reproducible and scalable approach for real-world applications.