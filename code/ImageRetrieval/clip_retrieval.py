import os
import torch
import pickle
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import clip
from flask import Flask, request, render_template, send_from_directory
from tqdm import tqdm
from werkzeug.utils import secure_filename

ROOT_DIR = "/home/suzana/Dokumente/uni/0_MA/MA/"  # Define the root directory


# Class to handle the loading and usage of the CLIP model
class CLIPService:
    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Use GPU if available
        self.model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device
        )  # Load the CLIP model

    # Function to extract features from an image using the CLIP model
    def get_image_features_database(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features.cpu().numpy().squeeze()

    # Another function to extract features from an image, with additional checks and preprocessing
    def get_image_features(self, image):
        try:
            image = Image.fromarray(image.astype("uint8"))
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
            return image_features.cpu().numpy().squeeze()
        except ValueError as e:
            print("Error occurred while processing image: ", e)
            print("Image shape: ", image.shape)
            print("Image data type: ", image.dtype)
            return None

    # Function to extract features from a text using the CLIP model
    def get_text_features(self, prompt):
        text = clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features.cpu().numpy()


# Class to handle the loading and saving of feature data
class FeatureDatabase:
    def __init__(self, clip_service, database_dir, features_dir, features_file_name):
        self.clip_service = clip_service
        self.database_dir = os.path.join(ROOT_DIR, database_dir)
        self.features_file = os.path.join(ROOT_DIR, features_dir, features_file_name)
        self.database = (
            self._load_or_generate_features()
        )  # Load or generate the feature database

    # Function to load precomputed features or generate them if they do not exist
    def _load_or_generate_features(self):
        if os.path.isfile(self.features_file):
            print("Loading precomputed features...")
            with open(self.features_file, "rb") as f:
                return pickle.load(f)
        else:
            print("Precomputing features...")
            database = []
            database_files = os.listdir(self.database_dir)
            for file in tqdm(database_files):
                file_path = os.path.join(self.database_dir, file)
                database.append(
                    (
                        file,
                        self.clip_service.get_image_features_database(
                            Image.open(file_path)
                        ),
                    )
                )
            with open(self.features_file, "wb") as f:
                pickle.dump(database, f)
            return database


# Class to handle the search of similar images or texts in the feature database
class SearchService:
    def __init__(self, clip_service, feature_database):
        self.clip_service = clip_service
        self.feature_database = feature_database

    # Function to calculate the cosine similarity between two feature vectors
    def calculate_similarity(self, features1, features2):
        return cosine_similarity(features1, features2)

    # Function to find similar images or texts in the feature database
    def find_similar_images(self, input_type, input_text, input_image):
        if input_type == "text":
            input_features = self.clip_service.get_text_features(input_text).squeeze()
        else:
            input_features = self.clip_service.get_image_features(input_image).squeeze()

        similarities = [
            self.calculate_similarity(
                input_features[np.newaxis, :], image_features[np.newaxis, :]
            )
            for filename, image_features in self.feature_database.database
        ]
        similarities = np.array(similarities).flatten()  # flatten the array
        indices = np.argsort(similarities)[::-1]
        sorted_database = [self.feature_database.database[i] for i in indices]
        sorted_similarities = [similarities[i] for i in indices]
        similar_images = [
            (filename, similarity)
            for (filename, _), similarity in zip(sorted_database, sorted_similarities)
        ]
        return similar_images


# Flask application to interact with the above services
class CLIPApp:
    def __init__(self, clip_service, feature_database, search_service):
        self.app = Flask(__name__)
        self.clip_service = clip_service
        self.feature_database = feature_database
        self.search_service = search_service

        self.app.route("/")(self.index)
        self.app.route("/search_text", methods=["POST"])(self.search_text)
        self.app.route("/search_image", methods=["POST"])(self.search_image)
        self.app.route("/<filename>")(self.send_file)

    # Function to run the Flask application
    def run(self, host="0.0.0.0", port=5000):
        self.app.run(debug=True, host=host, port=port)

    # Function to render the index page
    def index(self):
        return render_template("index.html")

    # Function to handle the search text route
    def search_text(self):
        text = request.form["text"]
        if text:  # check if text is not empty
            similar_images = self.search_service.find_similar_images("text", text, None)
            return render_template("similar_images.html", images=similar_images)
        else:
            print("No text input provided.")
            return render_template("index.html")

    # Function to handle the search image route
    def search_image(self):
        file = request.files["file"]
        filename = secure_filename(file.filename)
        if filename:  # Add this check
            file.save(os.path.join(ROOT_DIR, "code/ImageRetrieval/uploads", filename))
            image = Image.open(
                os.path.join(ROOT_DIR, "code/ImageRetrieval/uploads", filename)
            )
            similar_images = self.search_service.find_similar_images(
                "image", None, np.array(image)
            )
            return render_template("similar_images.html", images=similar_images)
        else:
            print("No file provided.")
            return render_template("index.html")

    # Function to send a file from the server
    def send_file(self, filename):
        return send_from_directory(
            directory=self.feature_database.database_dir,
            path=filename,
            as_attachment=True,
        )


# Main function to run the entire process
if __name__ == "__main__":
    clip_service = CLIPService()
    database_dir = "data/datasets/vismig"
    features_dir = "code/ImageRetrieval"
    features_file_name = "features.pkl"
    feature_database = FeatureDatabase(
        clip_service, database_dir, features_dir, features_file_name
    )
    search_service = SearchService(clip_service, feature_database)
    app = CLIPApp(clip_service, feature_database, search_service)
    app.run()
