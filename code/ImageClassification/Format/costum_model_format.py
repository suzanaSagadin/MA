import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from os.path import join
import torch.nn.functional as F
from shutil import copyfile
from datetime import datetime
import time

ROOT_DIR = "/home/suzana/Dokumente/uni/0_MA/MA"


def get_run_name(num_epochs):
    """
    Generate a unique name for each run based on current date and time.

    Args:
    num_epochs (int): Number of epochs in the current run.

    Returns:
    run_name (str): Unique name for the current run.
    """
    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M")
    run_name = f"E{num_epochs}_{date_time}"
    return run_name


class Net(nn.Module):
    """
    A PyTorch model implementing a Convolutional Neural Network (CNN).

    The architecture includes five convolutional layers each followed by a max pooling layer, a dropout layer, and three fully connected layers at the end.

    Attributes:
        conv1 to conv5: The convolutional layers of the network.
        pool1 to pool4: The max pooling layers of the network.
        dropout: A dropout layer for regularization.
        fc1 to fc3: The fully connected (dense) layers of the network.
    """

    def __init__(self, n_channels=3):
        """
        Initialize the CNN with the desired number of input channels.

        Parameters:
        n_channels: The number of input channels (default is 3, for RGB images).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 16, 5)  # Define the 1st convolutional layer
        self.pool1 = nn.MaxPool2d(2, 2)  # Define the 1st max pooling layer
        self.conv2 = nn.Conv2d(16, 32, 5)  # Define the 2nd convolutional layer
        self.pool2 = nn.MaxPool2d(2, 2)  # Define the 2nd max pooling layer
        self.conv3 = nn.Conv2d(32, 64, 3)  # Define the 3rd convolutional layer
        self.pool3 = nn.MaxPool2d(2, 2)  # Define the 3rd max pooling layer
        self.conv4 = nn.Conv2d(64, 128, 3)  # Define the 4th convolutional layer
        self.pool4 = nn.MaxPool2d(2, 2)  # Define the 4th max pooling layer
        self.conv5 = nn.Conv2d(128, 256, 3)  # Define the 5th convolutional layer
        self.dropout = nn.Dropout(0.5)  # Define the dropout

        # Dummy forward pass to calculate the number of features
        x = torch.randn(1, n_channels, 224, 224)
        self._to_linear = None
        self.convs(
            x
        )  # Run a dummy input through the convolutional layers to determine output shape

        self.fc1 = nn.Linear(
            self._to_linear, 1024
        )  # Define the 1st fully connected layer
        self.fc2 = nn.Linear(1024, 768)  # Define the 2nd fully connected layer
        self.fc3 = nn.Linear(768, 4)  # Define the 3rd fully connected layer

    def convs(self, x):
        """
        Define the forward pass through the convolutional part of the network.

        Parameters:
        x: Input data.

        Returns:
        x: Output after passing through the convolutional layers.
        """
        # Pass the input through each convolutional layer followed by a ReLU activation function and max pooling
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = F.relu(
            self.conv5(x)
        )  # Final convolutional layer does not have a max pooling

        # Calculate the total number of features coming out of the convolutional layers
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def forward(self, x):
        """
        Define the complete forward pass through the network.

        Parameters:
        x: Input data.

        Returns:
        x: Output after passing through the complete network.
        """
        x = self.convs(x)  # Pass input through the convolutional layers
        x = x.view(
            -1, self._to_linear
        )  # Flatten the output for the fully connected layers
        x = F.relu(self.fc1(x))  # Pass through the 1st fully connected layer
        x = F.relu(self.fc2(x))  # Pass through the 2nd fully connected layer
        x = self.fc3(x)  # Pass through the 3rd fully connected layer
        return x


class FormatDataset(Dataset):
    """
    A PyTorch Dataset for loading images and their corresponding labels from disk.

    This class inherits from the PyTorch Dataset class and overrides the __init__, __len__, and __getitem__ methods.
    It allows loading images and labels in a format that can be used by PyTorch's DataLoader for batching and shuffling.

    Attributes:
        image_paths: A list of paths to the image files.
        labels: A list of labels corresponding to each image.
        transform: Optional; a torchvision.transforms composition to be applied to the images.
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize the dataset with the image paths and labels.

        Parameters:
            image_paths: A list of paths to the image files.
            labels: A list of labels corresponding to each image.
            transform: Optional; a torchvision.transforms composition to be applied to the images.
        """
        self.image_paths = image_paths  # Store the image paths
        self.labels = labels  # Store the labels
        self.transform = transform  # Store the transform

    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
            The number of images in the dataset.
        """
        return len(self.image_paths)  # Return the number of image paths

    def __getitem__(self, idx):
        """
        Get the image and label at a given index.

        Parameters:
            idx: Index of the image and label to return.

        Returns:
            A tuple containing the image, label, and image path.
        """
        image_path = self.image_paths[idx]  # Get the image path at the provided index
        label = self.labels[idx]  # Get the corresponding label
        image = Image.open(image_path).convert(
            "RGB"
        )  # Open and convert the image to RGB
        if self.transform:  # If a transform is provided
            image = self.transform(image)  # Apply the transform
        return (
            image,
            label,
            image_path,
        )  # Return a tuple of the image, label, and image path


def train_model(model, criterion, optimizer, run_name, num_epochs=25):
    """
    Train a PyTorch model for a specified number of epochs.

    This function trains a model on a dataset, calculates training and validation accuracy and loss,
    and saves the model weights after training. It also generates and saves training and validation
    loss and accuracy curves.

    Parameters:
        model: The PyTorch model to be trained.
        criterion: The loss function.
        optimizer: The optimization algorithm.
        run_name: The name of the training run. Used in saving the training curves and model weights.
        num_epochs: The number of epochs for training the model (default is 25).
    """
    print(f"Starting model training for {num_epochs} epochs.")
    total_loss_train, total_acc_train = (
        [],
        [],
    )  # Lists to store training loss and accuracy values
    total_loss_val, total_acc_val = (
        [],
        [],
    )  # Lists to store validation loss and accuracy values

    start_time_train = time.time()  # Start time for training

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        model.train()  # Set model to training mode
        running_loss = 0.0  # Running loss for this epoch
        running_corrects = 0  # Running correct predictions for this epoch

        # Loop over data in the training DataLoader
        for inputs, labels, _ in tqdm(train_loader):
            inputs = inputs.to(device)  # Move inputs to device
            labels = labels.to(device)  # Move labels to device
            optimizer.zero_grad()  # Zero the parameter gradients

            # Forward pass with gradient calculation enabled
            with torch.set_grad_enabled(True):
                outputs = model(inputs)  # Forward pass
                _, preds = torch.max(outputs, 1)  # Get model predictions
                loss = criterion(outputs, labels)  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Optimize the model

            running_loss += loss.item() * inputs.size(0)  # Update running loss
            running_corrects += torch.sum(
                preds == labels.data
            )  # Update running correct predictions
        epoch_loss = running_loss / len(train_loader.dataset)  # Calculate epoch loss
        epoch_acc = running_corrects.double() / len(
            train_loader.dataset
        )  # Calculate epoch accuracy

        total_loss_train.append(epoch_loss)  # Append epoch loss to list
        total_acc_train.append(epoch_acc)  # Append epoch accuracy to list

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        model.eval()  # Set model to evaluation mode
        running_loss = 0.0  # Running loss for this epoch
        running_corrects = 0  # Running correct predictions for this epoch

        # Loop over data in the validation DataLoader
        for inputs, labels, _ in tqdm(val_loader):
            inputs = inputs.to(device)  # Move inputs to device
            labels = labels.to(device)  # Move labels to device
            optimizer.zero_grad()  # Zero the parameter gradients

            # Forward pass with gradient calculation disabled
            with torch.set_grad_enabled(False):
                outputs = model(inputs)  # Forward pass
                _, preds = torch.max(outputs, 1)  # Get model predictions
                loss = criterion(outputs, labels)  # Calculate loss

            running_loss += loss.item() * inputs.size(0)  # Update running loss
            running_corrects += torch.sum(
                preds == labels.data
            )  # Update running correct predictions

        epoch_loss = running_loss / len(val_loader.dataset)  # Calculate epoch loss
        epoch_acc = running_corrects.double() / len(
            val_loader.dataset
        )  # Calculate epoch accuracy

        total_loss_val.append(epoch_loss)  # Append epoch loss to list
        total_acc_val.append(epoch_acc)  # Append epoch accuracy to list

        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print()

        # Log training and validation loss and accuracy for this epoch
        logging.info(
            f"Epoch {epoch}, Train Loss: {total_loss_train[-1]}, Train Acc: {total_acc_train[-1]}, Val Loss: {total_loss_val[-1]}, Val Acc: {total_acc_val[-1]}"
        )

    end_time_train = time.time()  # End time for training
    elapsed_time_train = (
        end_time_train - start_time_train
    ) / 60  # Calculate elapsed time for training
    print(f"Time taken for training: {elapsed_time_train} minutes")
    logging.info(f"Time taken for training: {elapsed_time_train} minutes")

    epochs_range = range(num_epochs)

    # Plot and save training and validation loss and accuracy curves
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    ax[0].plot(epochs_range, total_loss_train, label="Train Loss")
    ax[0].plot(epochs_range, total_loss_val, label="Validation Loss")
    ax[0].legend(loc="upper right")
    ax[0].set_title("Train and Val Loss")
    ax[1].plot(epochs_range, total_acc_train, label="Train Accuracy")
    ax[1].plot(epochs_range, total_acc_val, label="Validation Accuracy")
    ax[1].legend(loc="lower right")
    ax[1].set_title("Train and Val Accuracy")
    fig.savefig(join("/", output_path, f"{run_name}_training_validation_curves.png"))

    # Save model weights after training
    torch.save(
        model.state_dict(),
        join("/", model_path, f"{run_name}_{i+1}_model_weights.pth"),
    )


def evaluate_model(model, loader, output_dir, criterion, run_name):
    """
    Evaluate a PyTorch model on a (test) dataset.

    This function evaluates a model on a dataset, calculates the loss and accuracy,
    and saves the images misclassified by the model into corresponding directories
    based on the misclassified label.

    Parameters:
        model: The PyTorch model to be evaluated.
        loader: DataLoader for the evaluation data.
        output_dir: The directory where the misclassified images will be saved.
        criterion: The loss function.
        run_name: The name of the training run. Used in naming the saved images.
    """

    print("Starting model evaluation.")

    # Create directories for misclassified images
    falseAs_label_0_dir = join(output_dir, f"false_class_as_{label_to_name[0]}")
    falseAs_label_1_dir = join(output_dir, f"false_class_as_{label_to_name[1]}")
    falseAs_label_2_dir = join(output_dir, f"false_class_as_{label_to_name[2]}")
    falseAs_label_3_dir = join(output_dir, f"false_class_as_{label_to_name[3]}")
    os.makedirs(falseAs_label_0_dir, exist_ok=True)
    os.makedirs(falseAs_label_1_dir, exist_ok=True)
    os.makedirs(falseAs_label_2_dir, exist_ok=True)
    os.makedirs(falseAs_label_3_dir, exist_ok=True)

    records = []  # To store records for CSV
    running_loss = 0.0  # Running loss for evaluation
    running_corrects = 0  # Running correct predictions for evaluation
    gt_label_counts = [0, 0, 0, 0]  # Counts for ground truth labels
    correct_label_counts = [0, 0, 0, 0]  # Counts for correct predictions per label

    with torch.no_grad():
        for inputs, labels, paths in tqdm(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Perform forward pass and get predictions
            outputs = model(inputs)
            outputs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            for label, pred, img_path, output in zip(labels, preds, paths, outputs):
                prob = output[pred].item()

                # Update counts for ground truth labels and correct predictions
                gt_label_counts[int(label.item())] += 1
                if int(label.item()) == int(pred.item()):
                    correct_label_counts[int(label.item())] += 1

                # Append record for CSV
                records.append(
                    {
                        "run-name": run_name,
                        "img-file-name": img_path,
                        "probability": float(prob),
                        "assigned-label": label_to_name[int(pred.item())],
                        "GT-label": label_to_name[int(label.item())],
                    }
                )

                # Save misclassified images into corresponding directories
                if label != 0 and pred == 0:
                    filename = (
                        f"{run_name}_{float(prob):.2f}_{os.path.basename(img_path)}"
                    )
                    copyfile(img_path, join(falseAs_label_0_dir, filename))

                elif label != 1 and pred == 1:
                    filename = (
                        f"{run_name}_{float(prob):.2f}_{os.path.basename(img_path)}"
                    )
                    copyfile(img_path, join(falseAs_label_1_dir, filename))

                elif label != 2 and pred == 2:
                    filename = (
                        f"{run_name}_{float(prob):.2f}_{os.path.basename(img_path)}"
                    )
                    copyfile(img_path, join(falseAs_label_2_dir, filename))

                elif label != 3 and pred == 3:
                    filename = (
                        f"{run_name}_{float(prob):.2f}_{os.path.basename(img_path)}"
                    )
                    copyfile(img_path, join(falseAs_label_3_dir, filename))

        # Calculate total loss and accuracy
        total_loss = running_loss / len(loader.dataset)
        total_acc = running_corrects.double() / len(loader.dataset)

        print(f"Test Loss: {total_loss:.4f} Acc: {total_acc:.4f}")

        for i in range(4):
            print(
                f"Label {label_to_name[i]} count: {gt_label_counts[i]} Correctly classified: {correct_label_counts[i]}"
            )

        # Log test loss and accuracy
        logging.info(f"Test Loss: {total_loss:.4f} Acc: {total_acc:.4f}")
        for i in range(4):
            logging.info(
                f"Label {label_to_name[i]} count: {gt_label_counts[i]} Correctly classified: {correct_label_counts[i]}"
            )
        logging.info(
            "-----------------------------------------------------------------------"
        )

    # Save records into CSV
    df = pd.DataFrame(records)
    if os.path.isfile(join(output_dir, "model_evaluation.csv")):
        df.to_csv(
            join(output_dir, "model_evaluation.csv"),
            mode="a",
            header=False,
            index=False,
        )
    else:
        df.to_csv(join(output_dir, "model_evaluation.csv"), index=False)

    print("Model evaluation complete. Results saved to CSV.")


if __name__ == "__main__":
    # Number of runs for the program
    num_runs = 1
    for i in range(num_runs):
        print(f"Starting process run {i+1}.")

        # Define input, output and model paths
        input_path = join(ROOT_DIR, "data/datasets/format/")
        output_path = join(ROOT_DIR, "code/ImageClassification/Format/eval/")
        model_path = join(ROOT_DIR, "code/ImageClassification/Models/Format/")

        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        print(f"Model path: {model_path}")

        # Create output and model directories if they don't exist
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        # Set up logging to a file
        logging.basicConfig(
            filename=join("/", output_path, "training.log"), level=logging.INFO
        )
        print("Logging setup done.")
        logging.info(f"Process run: {i+1} of: {num_runs}")

        # Exclude certain labels and list remaining labels in directory
        labels_to_exclude = ["Visual art", "Unknown", "Print media", "Print materials"]
        labels = [
            label for label in os.listdir(input_path) if label not in labels_to_exclude
        ]
        labels.sort()  # ensure the same order
        label_to_name = {i: name for i, name in enumerate(labels)}
        print(f"Labels to name mapping: {label_to_name}")

        # Collect all image data and their corresponding labels
        data = []
        for l, label in enumerate(labels):
            path = join(input_path, label)
            for img in os.listdir(path):
                data.append((f"{path}/{img}", l))

        print("Data loaded.")

        # Define transformations for the image data
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Instantiate the model and move it to GPU if available
        model = Net(n_channels=3)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Set up loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create dataset and data loaders for train, validation, and test sets
        full_dataset = FormatDataset(*zip(*data), transform=transform)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.2 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Train and evaluate the model
        num_epochs = 25
        run_name = get_run_name(num_epochs)
        train_model(model, criterion, optimizer, run_name, num_epochs=num_epochs)
        evaluate_model(model, test_loader, output_path, criterion, run_name)

        print(f"Procces run {i+1} completed.")
