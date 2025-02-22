import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import numpy as np

CIFAR_CLASS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']
   
def load_dataset(batch_size = 32):
    # Define transformation for normalization and resizing
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # resizes to 224x224 px
        transforms.ToTensor(), # converts imgs from numpy or pil img to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #normalise img using mean and std for imagenet. resnet was trained on imagenet and expects similar normalization
    ])
    
    # Loads the train and test dataset
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Splits the training dataset into train and val sets (80% train, 20% val)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Create DataLoaders for train, val and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
       
    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader

def train(model, train_loader, len_trainset, val_loader, len_valset, loss_function, optimizer, device):
    # Set the model to train mode
    model.train()

    # Initialize the running loss and accuracy
    running_loss = 0.0
    running_corrects = 0

    # Iterate over the batches of the train loader
    for inputs, labels in train_loader:
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the optimizer gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = loss_function(outputs, labels)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Update the running loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # Calculate the train loss and accuracy
    train_loss = running_loss / len_trainset
    train_acc = running_corrects.double() / len_trainset
    
    # Set the model to evaluation mode
    model.eval()

    # Initialize the running loss and accuracy
    running_loss = 0.0
    running_corrects = 0

    # Iterate over the batches of the validation loader
    for inputs, labels in val_loader:
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = loss_function(outputs, labels)

        # Update the running loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # Calculate the validation loss and accuracy
    val_loss = running_loss / len_valset
    val_acc = running_corrects.double() / len_valset
    
    return model, train_loss, train_acc, val_loss, val_acc

def test(model, testloader, len_testset, loss_function, filename, device="cuda"):
    print(f"Testing the model")
    test_loss = 0.0
    test_total = 0
    test_correct = 0

    # Switch to evaluation mode
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            # Update test loss
            test_loss += loss.item() * inputs.size(0)

            # Compute test accuracy
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    compute_metrics(all_preds, all_labels, filename)

    # Compute average test loss and accuracy
    test_loss = test_loss / len_testset
    test_accuracy = 100.0 * test_correct / test_total

    return test_loss, test_accuracy

def compute_metrics(preds, labels, filename):
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    
    conf_matrix = confusion_matrix(labels, preds)
    
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [accuracy, precision, recall, f1]
    })
    
    metrics_csv_path = f"results/{filename}_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CIFAR_CLASS, yticklabels=CIFAR_CLASS)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {filename}")
    conf_matrix_path = f"results/{filename}_confusion_matrix.png"
    plt.savefig(conf_matrix_path)
    print(f"Confusion matrix saved to {conf_matrix_path}")
    plt.close()
    
def save_sample_predictions(model, test_loader, device, filename_prefix, num_samples=10):
    """Saves a sample of model test results with actual vs. predicted labels."""

    model.eval()  # Set model to evaluation mode
    images, actual_labels, predicted_labels = [], [], []

    transform = transforms.ToPILImage()  # Convert tensor to image
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Store the first `num_samples` test images and their labels
            for i in range(min(num_samples, inputs.shape[0])):
                images.append(transform(inputs[i].cpu()))  # Convert tensor to PIL image
                actual_labels.append(CIFAR_CLASS[labels[i].item()])
                predicted_labels.append(CIFAR_CLASS[preds[i].item()])
            
            if len(images) >= num_samples:
                break  # Stop after collecting enough samples

    # Ensure results directory exists
    os.makedirs("results/test_samples", exist_ok=True)

    # Plot and save the results
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))  # Create a 2x5 grid for 10 images
    fig.suptitle("Model Test Predictions", fontsize=14)

    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            ax.imshow(images[idx])
            ax.set_title(f"Actual: {actual_labels[idx]}\nPred: {predicted_labels[idx]}", fontsize=10, color="green" if actual_labels[idx] == predicted_labels[idx] else "red")
            ax.axis("off")
    
    sample_results_path = f"results/{filename_prefix}_sample_predictions.png"
    plt.savefig(sample_results_path)  # Save figure
    plt.show()
    print(f"Sample test predictions saved at {sample_results_path}")       
    
    