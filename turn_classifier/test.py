import torch
from road_classifier import load_model
from utils import load_data
import dense_transforms
import numpy as np
from glob import glob
from os import path
from PIL import Image
import torchvision.transforms.functional as TF

DATASET_PATH = 'validation_road_data'


# Dataset for validation data
class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        self.data = []
        self.transform = transform

        # Load all image paths and labels
        for f in glob(path.join(dataset_path, '*.png')):
            img = Image.open(f).convert("RGB")
            img.load()
            label = int(f.split('/')[1][0])  # Extract label from filename
            self.data.append((img, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        # Apply the transformation
        if self.transform:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)
        label = torch.tensor(label, dtype=torch.long)
        return img[0], label  # Return as (image, label)


# Function to compute accuracy
def compute_accuracy(predictions, labels):
    """
    Custom function to compute accuracy.
    Args:
        predictions (list): Predicted class indices.
        labels (list): True class indices.

    Returns:
        float: Accuracy as a percentage.
    """
    correct = sum(p == l for p, l in zip(predictions, labels))
    total = len(labels)
    accuracy = correct / total
    return accuracy


# Function to test the model
def test_model(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculations for testing
        for img, label in data_loader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            _, predicted = torch.max(pred, dim=1)  # Get the predicted class (logits to class index)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            print('Predicted:')
            print(predicted)
            print('---------')
            print('Labels:')
            print(label, end='\n\n')
            #print('prediction: ' + predicted + ' | ' + 'label: ' + str(label.item()))
    # Compute accuracy using the custom function
    accuracy = compute_accuracy(all_preds, all_labels)
    return accuracy


if __name__ == "__main__":
    # Load the model
    model = load_model()
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model = model.to(device)

    # Set up validation dataset and DataLoader
    transform = dense_transforms.ToTensor()
    validation_data = ValidationDataset(DATASET_PATH, transform=transform)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=16, shuffle=False)

    # Test the model
    accuracy = test_model(model, validation_loader, device)
    print(f"Model Accuracy on Validation Set: {accuracy * 100:.2f}%")

