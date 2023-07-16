# Import necessary libraries
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler
import pandas as pd
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
from os.path import join
from shutil import copyfile
from datetime import datetime
import time

# Define root directory for data
ROOT_DIR = "/home/suzana/Dokumente/uni/0_MA/MA"


# Function to create a unique identifier for each run
def get_run_name(num_epochs):
    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M")
    run_name = f"E{num_epochs}_{date_time}"
    return run_name


# Define the network architecture
class Net(nn.Module):
    def __init__(self, n_channels=3):
        super().__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(n_channels, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.dropout = nn.Dropout(0.5)  # Dropout layer

        # Dummy forward pass to calculate the number of features
        x = torch.randn(1, n_channels, 224, 224)
        self._to_linear = None
        self.convs(x)

        # Define the fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 768)
        self.fc3 = nn.Linear(768, 1)

    # Define the forward pass through the convolutional layers
    def convs(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    # Define the forward pass through the entire network
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = self.dropout(F.relu(self.fc1(x)))  # Apply dropout after ReLU
        x = self.dropout(F.relu(self.fc2(x)))  # Apply dropout after ReLU
        x = self.fc3(x)
        return x


# Define a custom dataset for the images
class KeywordsDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        transform=None,
        augment_transform=None,
        augment_label=0,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment_transform = augment_transform
        self.augment_label = augment_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.augment_transform and label == self.augment_label:
            image = self.augment_transform(image)
        return image, torch.tensor(label, dtype=torch.float32), image_path


# Function to train the model
def train_model(model, criterion, optimizer, run_name, num_epochs=25):
    print(f"Starting model training for {num_epochs} epochs.")
    total_loss_train, total_acc_train = [], []
    total_loss_val, total_acc_val = [], []

    # Start timing
    start_time_train = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels, _ in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.view(-1, 1)  # reshape the labels
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                preds = torch.sigmoid(outputs) >= 0.5  # apply sigmoid and threshold
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        total_loss_train.append(epoch_loss)
        total_acc_train.append(epoch_acc)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels, _ in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.view(-1, 1)  # reshape the labels
            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = torch.sigmoid(outputs) >= 0.5  # apply sigmoid and threshold
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        total_loss_val.append(epoch_loss)
        total_acc_val.append(epoch_acc)

        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        logging.info(
            f"Epoch {epoch}, Train Loss: {total_loss_train[-1]}, Train Acc: {total_acc_train[-1]}, Val Loss: {total_loss_val[-1]}, Val Acc: {total_acc_val[-1]}"
        )

    end_time_train = time.time()
    elapsed_time_train = (end_time_train - start_time_train) / 60
    print(f"Time taken for training: {elapsed_time_train} minutes")
    logging.info(f"Time taken for training: {elapsed_time_train} minutes")

    # Plot training and validation curves
    epochs_range = range(num_epochs)
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    ax[0].plot(epochs_range, total_loss_train, label="Train Loss")
    ax[0].plot(epochs_range, total_loss_val, label="Validation Loss")
    ax[0].legend(loc="upper right")
    ax[0].set_title("Train and Val Loss")

    ax[1].plot(epochs_range, total_acc_train, label="Train Accuracy")
    ax[1].plot(epochs_range, total_acc_val, label="Validation Accuracy")
    ax[1].legend(loc="lower right")
    ax[1].set_title("Train and Val Accuracy")

    # Save the plot to a PNG file
    fig.savefig(join("/", output_path, f"{run_name}_training_validation_curves.png"))

    torch.save(
        model.state_dict(), join("/", model_path, f"{run_name}_{i+1}_model_weights.pth")
    )


# Function to evaluate the model
def evaluate_model(model, loader, output_dir, criterion, run_name):
    print("Starting model evaluation.")
    run_name = get_run_name(num_epochs)
    false_positives_dir = join(
        output_dir, f"false_classification_as_{label_to_name[1]}"
    )
    false_negatives_dir = join(
        output_dir, f"false_classification_as_{label_to_name[0]}"
    )
    os.makedirs(false_positives_dir, exist_ok=True)
    os.makedirs(false_negatives_dir, exist_ok=True)

    # Initialize a list to store the records
    records = []

    # Initialize variables for loss, accuracy and label counts
    running_loss = 0.0
    running_corrects = 0
    gt_label_counts = [0, 0]  # [label-1, label-2]
    correct_label_counts = [0, 0]  # [Correct label-1, Correct label-2]

    with torch.no_grad():
        for inputs, labels, paths in tqdm(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.view(-1, 1)  # reshape the labels

            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Calculate loss

            preds = torch.sigmoid(outputs) >= 0.5  # apply sigmoid and threshold

            running_loss += loss.item() * inputs.size(0)  # Add the batch loss
            running_corrects += torch.sum(
                preds == labels.data
            )  # Add the correct predictions

            for label, pred, img_path, output in zip(labels, preds, paths, outputs):
                prob = (
                    torch.sigmoid(output).item()
                    if pred == 1
                    else 1 - torch.sigmoid(output).item()
                )

                # Count GT-labels and correct predictions
                gt_label_counts[int(label.item())] += 1
                if int(label.item()) == int(pred.item()):
                    correct_label_counts[int(label.item())] += 1

                # Add the record to the list
                records.append(
                    {
                        "run-name": run_name,
                        "img-file-name": img_path,
                        "probability": float(prob),
                        "assigned-label": label_to_name[int(pred.item())],
                        "GT-label": label_to_name[int(label.item())],
                    }
                )

                if label == 0 and pred == 1:  # False Positive
                    filename = (
                        f"{run_name}_{float(prob):.2f}_{os.path.basename(img_path)}"
                    )
                    copyfile(
                        img_path, join(false_positives_dir, filename)
                    )  # Moved to false_positives
                elif label == 1 and pred == 0:  # False Negative
                    filename = (
                        f"{run_name}_{float(prob):.2f}_{os.path.basename(img_path)}"
                    )
                    copyfile(
                        img_path, join(false_negatives_dir, filename)
                    )  # Moved to false_negatives

        # Calculate total loss and accuracy
        total_loss = running_loss / len(loader.dataset)
        total_acc = running_corrects.double() / len(loader.dataset)

        print(f"Test Loss: {total_loss:.4f} Acc: {total_acc:.4f}")
        print(
            f"Label {label_to_name[0]} count: {gt_label_counts[0]} Correctly classified: {correct_label_counts[0]}"
        )
        print(
            f"Label {label_to_name[1]} count: {gt_label_counts[1]} Correctly classified: {correct_label_counts[1]}"
        )

        # add to log
        logging.info(f"Test Loss: {total_loss:.4f} Acc: {total_acc:.4f}")
        logging.info(
            f"Label {label_to_name[0]} count: {gt_label_counts[0]} Correctly classified: {correct_label_counts[0]}"
        )
        logging.info(
            f"Label {label_to_name[1]} count: {gt_label_counts[1]} Correctly classified: {correct_label_counts[1]}"
        )

        logging.info(
            "-----------------------------------------------------------------------"
        )

    # Convert the list to a DataFrame
    df = pd.DataFrame(records)

    # Save the DataFrame to a CSV file
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


# Main function to run the entire process
if __name__ == "__main__":
    num_runs = 1
    for i in range(num_runs):
        print(f"Starting process run {i+1}.")

        # Keyword and paths
        keyword = "S025"

        input_path = join(ROOT_DIR, f"data/datasets/keywords/{keyword}")
        output_path = join(
            ROOT_DIR, f"code/ImageClassification/keywords/eval/{keyword}"
        )
        model_path = join(
            ROOT_DIR, f"code/ImageClassification/Models/keywords/{keyword}"
        )

        print(f'Setup for keyword "{keyword}"')
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        print(f"Model path: {model_path}")
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            filename=join("/", output_path, "training.log"), level=logging.INFO
        )
        print("Logging setup done.")

        logging.info(f"Process run: {i+1} of: {num_runs}")

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        augment_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
            ]
        )

        # Instantiate the model
        model = Net(n_channels=3)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print("Model instantiated and moved to device.")

        # Load data
        labels = os.listdir(input_path)
        labels.sort()  # ensure the same order
        label_to_name = {i: name for i, name in enumerate(labels)}
        print(f"Labels to name mapping: {label_to_name}")
        data = []

        for l, label in enumerate(labels):
            path = join(input_path, label)
            for img in os.listdir(path):
                data.append((join(path, img), l))

        print("Data loaded.")

        # Create your dataset
        full_dataset = KeywordsDataset(
            *zip(*data),
            transform=transform,
            augment_transform=augment_transform,
            augment_label=0,
        )

        # Create train, validation and test datasets
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.2 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        # Compute samples weights only for the training set
        train_data = [data[i] for i in train_dataset.indices]
        class_sample_counts_train = np.bincount([label for (_, label) in train_data])
        class_weights_train = 1.0 / torch.tensor(
            class_sample_counts_train, dtype=torch.float
        )
        samples_weights_train = class_weights_train[
            [label for (_, label) in train_data]
        ]

        # Create the sampler
        sampler = WeightedRandomSampler(
            weights=samples_weights_train,
            num_samples=len(samples_weights_train),
            replacement=True,
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        num_epochs = 25
        run_name = get_run_name(num_epochs)

        train_model(model, criterion, optimizer, run_name, num_epochs=num_epochs)

        evaluate_model(model, test_loader, output_path, criterion, run_name)

        print(f"Procces run {i+1} completed.")
