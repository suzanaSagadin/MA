import os
import shutil
import pandas as pd
import ast
from sklearn.metrics import accuracy_score
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    filename="labels_search.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Define root directory and all other paths relative to that
ROOT_DIR = "/home/suzana/Dokumente/uni/0_MA/MA/"
CSV1_PATH = os.path.join(
    ROOT_DIR,
    "code/ObjectDetection/COCO-PanopticSegmentation_panoptic_fpn_R_101_3x_results.csv",
)
CSV2_PATH = os.path.join(
    ROOT_DIR,
    "code/ObjectDetection/LVISv0.5-InstanceSegmentation_mask_rcnn_X_101_32x8d_FPN_1x_results.csv",
)
CSV3_PATH = os.path.join(ROOT_DIR, "data/datasets/keyword_500_label_data.csv")
BASE_DIR = os.path.join(ROOT_DIR, "code/ObjectDetection/label_search/")
os.makedirs(BASE_DIR, exist_ok=True)

FALSE_CLASS_AS_500_DIR = os.path.join(BASE_DIR, "false_class_as_500")
FALSE_CLASS_AS_OTHER_DIR = os.path.join(BASE_DIR, "false_class_as_other")
CORRECT_CLASS_AS_500_DIR = os.path.join(BASE_DIR, "correct_class_as_500")
os.makedirs(FALSE_CLASS_AS_500_DIR, exist_ok=True)
os.makedirs(FALSE_CLASS_AS_OTHER_DIR, exist_ok=True)
os.makedirs(CORRECT_CLASS_AS_500_DIR, exist_ok=True)

# Define labels to search for
SEARCH_LABELS = [
    "airplane",
    "boat",
    "buoy",
    "kayak",
    "mast",
    "oar",
    "propeller",
    "sail",
]

# Define the label name
LABEL_NAME = "500"


# Function to read and process the CSV files
def read_and_process_csvs():
    # Read the CSV files
    df1 = pd.read_csv(CSV1_PATH)
    df2 = pd.read_csv(CSV2_PATH)

    # Convert the strings into lists
    df1["Classes"] = df1["Classes"].apply(ast.literal_eval)
    df2["Classes"] = df2["Classes"].apply(ast.literal_eval)

    # Merge the dataframes on the 'Image_Path' column
    merged_df = pd.merge(df1, df2, on="Image_Path")

    return merged_df


# Function to get unique values from the dataframe
def get_unique_values(merged_df):
    # Explode the list columns into multiple rows
    classes_x_values = merged_df["Classes_x"].explode().dropna().unique()
    classes_y_values = merged_df["Classes_y"].explode().dropna().unique()

    # Combine and get unique values from both
    combined_classes = np.concatenate((classes_x_values, classes_y_values), axis=None)
    unique_combined_classes = np.unique(combined_classes)

    return unique_combined_classes


# Function to extract and copy images based on their classification
def extract_and_copy_images(merged_df):
    positive_dir = os.path.join(BASE_DIR, LABEL_NAME)
    os.makedirs(positive_dir, exist_ok=True)

    # Define images that contain the search labels
    positive_images = merged_df[
        merged_df["Classes_x"].apply(lambda x: bool(set(x) & set(SEARCH_LABELS)))
        | merged_df["Classes_y"].apply(lambda x: bool(set(x) & set(SEARCH_LABELS)))
    ].copy()
    positive_images["label"] = LABEL_NAME

    # Define images that do not contain the search labels
    negative_images = merged_df[
        ~merged_df["Image_Path"].isin(positive_images["Image_Path"])
    ].copy()
    negative_images["label"] = "other"

    # Combine positive and negative images
    final_df = pd.concat([positive_images, negative_images])

    # Copy images to the positive directory
    for image_path in positive_images["Image_Path"]:
        shutil.copy(image_path, positive_dir)

    return final_df


# Function to save the final dataframe to a CSV file
def save_to_csv(final_df):
    final_df.to_csv(
        os.path.join(BASE_DIR, f"{LABEL_NAME}_final.csv"),
        index=False,
    )


# Function to compute accuracy of the predictions
def compute_accuracy(final_df):
    df3 = pd.read_csv(CSV3_PATH)

    df3["Image_Path"] = df3["img"]
    df3.drop(["collection", "img"], axis=1, inplace=True)

    final_df.drop(["Classes_x", "Classes_y"], axis=1, inplace=True)
    final_df["Image_Path"] = final_df["Image_Path"].str.replace(
        os.path.join(ROOT_DIR, "data/datasets/keywords/500/other/"), ""
    )
    final_df["Image_Path"] = final_df["Image_Path"].str.replace(
        os.path.join(ROOT_DIR, "data/datasets/keywords/500/500/"), ""
    )

    # Merge the final dataframe with the ground truth labels
    comparison_df = pd.merge(
        final_df, df3, on="Image_Path", suffixes=("_predicted", "_actual")
    )

    # Compute accuracy
    accuracy = accuracy_score(
        comparison_df["label_predicted"], comparison_df["label_actual"]
    )
    logging.info(f"Accuracy: {accuracy*100:.2f}%")

    return accuracy, comparison_df


# Function to handle misclassified images
def handle_misclassifications(comparison_df):
    false_class_as_500 = 0
    false_class_as_other = 0
    correct_class_as_500 = 0

    # For each row in the comparison dataframe
    for index, row in comparison_df.iterrows():
        img_path = os.path.join(
            ROOT_DIR,
            f"data/datasets/keywords/500/{row['label_actual']}/{row['Image_Path']}",
        )

        # Check if the predicted label matches the actual label
        if row["label_predicted"] != row["label_actual"]:
            if row["label_predicted"] == "500":
                false_class_as_500 += 1
                shutil.copy(img_path, FALSE_CLASS_AS_500_DIR)
            else:
                false_class_as_other += 1
                shutil.copy(img_path, FALSE_CLASS_AS_OTHER_DIR)
        elif row["label_predicted"] == "500":
            correct_class_as_500 += 1
            shutil.copy(img_path, CORRECT_CLASS_AS_500_DIR)

    # Log the number of correctly and incorrectly classified images
    logging.info(
        f"Number of images correctly classified as 500: {correct_class_as_500}"
    )
    logging.info(f"Number of images misclassified as 500: {false_class_as_500}")
    logging.info(f"Number of images misclassified as 'other': {false_class_as_other}")


# Main function to run the entire process
if __name__ == "__main__":
    logging.info("Starting program.")

    merged_df = read_and_process_csvs()
    final_df = extract_and_copy_images(merged_df)
    unique_labels = get_unique_values(merged_df)
    logging.info(f"Search labels: {SEARCH_LABELS}")
    logging.info(f"Total number of images: {len(final_df)}")

    save_to_csv(final_df)
    accuracy, comparison_df = compute_accuracy(final_df)
    print(f"Accuracy: {accuracy*100:.2f}%")
    handle_misclassifications(comparison_df)

    logging.info("Program completed.")
