# README

This code repository is a part of the master's thesis titled "Einsatz von Computer-Vision-Methoden zur Analyse historischer Fotobestände" for the Master of Arts (MA) degree at the Karl-Franzens-Universität Graz. The project aims to investigate various Computer Vision methods for analyzing historical photograph collections.

**Title of Thesis:**  
Einsatz von Computer-Vision-Methoden zur Analyse historischer Fotobestände

**Subtitle:**  
Forschungsstand, Möglichkeiten und Grenzen

**University:**  
Karl-Franzens-Universität Graz

**Author:**  
Suzana SAGADIN

**Supervisor:**  
Mag. Dr. phil. Martina Scholger

**Institution:**  
Institut Zentrum für Informationsmodellierung – Austrian Centre for Digital Humanities

**Year:**  
2023


---

## About the repo

This repo contains Python code for training and evaluating machine learning models for various tasks including clustering, image classification, and object detection. The project is structured as follows:

## Main Directory Structure

```
.
├── code
│   ├── Clustering
│   ├── ImageClassification
│   ├── ImageRetrieval
│   └── ObjectDetection
└── data
    ├── corpus
    └── datasets
```

## Corpus

The corpus for this project can be found on the Y drive in the temp folder. Please copy the corpus to the `data/corpus/img` directory before running the scripts.

## Scripts (data directory)

1. `create_format_dataset.py`: Assigns labels to images in the dataset and organizes them into directories based on the labels.
2. `create_keyword_dataset.py`: Assigns labels to images based on the presence of specific keywords and organizes them into directories based on the labels.
3. `create_vismig_dataset.py`: Creates a dataset specific to the "vismig" category by copying images from the "corpus" directory.
4. `eval_corpus.py`: Evaluates the corpus by performing specific operations on the dataset.
5. `helper_functions.py`: Contains helper functions used by other scripts.
6. main.py: Main function to run the entire process

## Scripts (code directory)

This project includes the following Python scripts:

1. `costum_model_format.py`: Trains and evaluates a custom CNN model on an image classification task with a specific format.
2. `costum_model_keywords.py`: Trains and evaluates a custom CNN model on an image classification task for specific keywords.
3. `detection.py`: Performs object detection on images using pre-trained Detectron2 models and writes the results to a CSV file.
4. `labels_search.py`: Processes the results of two object detection models and calculates the accuracy of their predictions.
5. `clip_retrieval.py`: Implements image retrieval using the CLIP model.
6. `clip_retrieval_S002.py`: Implements image retrieval using the CLIP model with modifications for the S002 dataset.
7. `k-means.py`: Performs k-means clustering on the dataset.

## How to Run

To run the Python scripts, you will need Python 3.x and the required packages installed. You can install the necessary packages using pip:

```
pip install -r requirements.txt
```

Note that `cv2` and `detectron2` are not standard Python packages and they require additional steps to install. For `cv2`, you might need to install `opencv-python` package. And `detectron2` installation can be complex and depends on the system.

Then, you can run the Python scripts as follows:

```
python3 script_name.py
```

Replace `script_name.py` with the name of the script you wish to run.

Please ensure that you create the datasets first before running the experiments in the `code` folder.

All paths in Python files need to be changed to your own, according to your local setup.
