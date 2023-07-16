# Import necessary libraries
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

ROOT_DIR = "/home/suzana/Dokumente/uni/0_MA/MA"  # Root directory for the project

# Set up logging
logging.basicConfig(filename="clustering.log", level=logging.INFO)


# Function to create a directory if it does not exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Function to load the pre-trained ResNet50 model
def load_model():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = model.eval()
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the last layer
    return model


# Function to transform the input image
def transform_image():
    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda image: image.convert("RGB")
            ),  # Convert image to RGB
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),  # Convert to tensor
        ]
    )
    return transform


# Function to extract features from the images using the pre-trained ResNet50 model
def extract_features(model, transform, path):
    features = []
    filenames = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            try:
                with Image.open(os.path.join(path, file)) as img:
                    img_t = transform(img)
                    img_u = torch.unsqueeze(img_t, 0)
                    feature = model(img_u)
                    feature = feature.detach().numpy().flatten()
                    features.append(feature)
                    filenames.append(file)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    return features, filenames


# Function to perform PCA on the extracted features
def perform_pca(features, use_fixed_random_state, fixed_random_state=None):
    if use_fixed_random_state:
        pca = PCA(n_components=100, random_state=fixed_random_state)
    else:
        pca = PCA(n_components=100)
    pca.fit(features)
    features_pca = pca.transform(features)
    return features_pca


# Function to determine the number of clusters using the elbow method
def determine_clusters(features_pca):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(
            n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0
        )
        kmeans.fit(features_pca)
        wcss.append(kmeans.inertia_)
    return wcss


# Function to plot the elbow graph
def plot_elbow_graph(wcss, run):
    plt.plot(range(1, 11), wcss)
    plt.title("The Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.savefig(os.path.join(ROOT_DIR, f"code/Clustering/elbow_{run}.png"))


# Function to perform K-means clustering on the PCA-transformed features
def perform_kmeans(
    features_pca, optimal_num_clusters, use_fixed_random_state, fixed_random_state=None
):
    if use_fixed_random_state:
        kmeans = KMeans(
            n_clusters=optimal_num_clusters, random_state=fixed_random_state
        )
    else:
        kmeans = KMeans(n_clusters=optimal_num_clusters)
    kmeans.fit(features_pca)
    return kmeans


# Function to save the cluster data
def save_cluster_data(filenames, kmeans, optimal_num_clusters, dataset, run):
    df = pd.DataFrame({"filename": filenames, "cluster": kmeans.labels_})
    df.to_csv(
        os.path.join(
            ROOT_DIR,
            f"code/Clustering/{dataset}_clusters_{optimal_num_clusters}_run_{run}_rs{use_fixed_random_state}.csv",
        ),
        index=False,
    )


# Function to save the images into their respective cluster directories
def save_cluster_images(path, output_path, filenames, kmeans):
    for file, cluster in zip(filenames, kmeans.labels_):
        cluster_dir = os.path.join(output_path, str(cluster))
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        img = Image.open(os.path.join(path, file))
        img.save(os.path.join(cluster_dir, file))


# Main function to run the entire process
if __name__ == "__main__":
    # Set parameters
    dataset_name = "vismig"
    runs = 1  # The number of runs
    optimal_num_clusters = (
        5  # Set this to a chosen number of clusters (best based on the Elbow method)
    )
    use_fixed_random_state = True  # Change this to True to use a fixed random state
    fixed_random_state = (
        22  # The fixed random state to use if use_fixed_random_state is True
    )

    # Log the start of the process
    logging.info(f"random_state: {use_fixed_random_state}")
    if use_fixed_random_state:
        logging.info(f"fixed_random_state: {fixed_random_state}")

    for run in range(1, runs + 1):
        logging.info(
            f"Starting run number {run} for dataset {dataset_name} with {optimal_num_clusters} clusters"
        )

        path = os.path.join(ROOT_DIR, f"data/datasets/{dataset_name}")
        output_path = os.path.join(
            ROOT_DIR,
            f"code/Clustering/{dataset_name}_clusters_{optimal_num_clusters}_run_{run}_rs{use_fixed_random_state}",
        )
        create_directory(output_path)
        model = load_model()
        transform = transform_image()
        features, filenames = extract_features(model, transform, path)

        if not features:
            print("No features were extracted. Check your images.")

        logging.info(f"Extracted {len(features)} features")

        features = np.array(features)
        features_pca = perform_pca(features, use_fixed_random_state, fixed_random_state)
        # wcss = determine_clusters(features_pca)
        # plot_elbow_graph(wcss, run)
        kmeans = perform_kmeans(
            features_pca,
            optimal_num_clusters,
            use_fixed_random_state,
            fixed_random_state,
        )
        save_cluster_data(filenames, kmeans, optimal_num_clusters, dataset_name, run)
        save_cluster_images(path, output_path, filenames, kmeans)

        # Logging the number of images in each cluster
        cluster_counts = pd.Series(kmeans.labels_).value_counts().to_dict()
        logging.info(f"Number of images in each cluster: {cluster_counts}")
        logging.info("------------------------------------------------------")
