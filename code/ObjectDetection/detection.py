# necessary imports
import os
import cv2
import csv
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
import torch
from tqdm import tqdm

# Define the root directory
ROOT_DIR = "/home/suzana/Dokumente/uni/0_MA/MA/"

# Dictionaries of models with their configurations and weights
MODELS = {
    # Define model configurations and weights in a tuple for each model
    "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x": (
        os.path.join(
            ROOT_DIR,
            ".venv/lib/python3.11/site-packages/detectron2/model_zoo/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml",
        ),
        "detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl",
    ),
    "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x": (
        os.path.join(
            ROOT_DIR,
            ".venv/lib/python3.11/site-packages/detectron2/model_zoo/configs/LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml",
        ),
        "detectron2://LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x/144219108/model_final_5e3439.pkl",
    ),
}


def configure_model(model_name):
    """
    Configure the model for prediction.

    :param model_name: str, name of the model
    :return: Detectron2 configuration object with the specified model's configuration
    """
    cfg = get_cfg()  # get a fresh new default config
    cfg.merge_from_file(MODELS[model_name][0])  # merge the config file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        0.5  # set the testing threshold for this model
    )
    cfg.MODEL.WEIGHTS = MODELS[model_name][1]  # specify model weights
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # decide the device for processing
    cfg.MODEL.DEVICE = device  # set the device
    return cfg


def run_detection(cfg, image_path):
    """
    Perform object detection on an image.

    :param cfg: Detectron2 configuration object
    :param image_path: str, path to the image
    :return: Detectron2 outputs object containing the result
    """
    predictor = DefaultPredictor(cfg)  # initialize the predictor
    im = cv2.imread(image_path)  # read the image
    outputs = predictor(im)  # perform object detection
    return outputs


def write_results_to_csv(output_file, image_path, outputs, metadata):
    """
    Write the detection results to a CSV file.

    :param output_file: str, path to the output file
    :param image_path: str, path to the image
    :param outputs: Detectron2 outputs object containing the result
    :param metadata: Metadata associated with the dataset used for detection
    """
    with open(output_file, "a", newline="") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",")  # initialize the CSV writer
        class_indices = (
            outputs["instances"].pred_classes.cpu().numpy()
        )  # extract class indices
        class_names = [
            metadata.thing_classes[i] for i in class_indices
        ]  # convert class indices to names
        filewriter.writerow(
            [image_path, class_names]
        )  # write the image path and detected classes to the CSV file


if __name__ == "__main__":
    image_dir = os.path.join(
        ROOT_DIR, "data/datasets/keywords/500/500"
    )  # directory containing the images
    output_dir = os.path.join(
        ROOT_DIR, "code/ObjectDetection/"
    )  # directory to save the output

    # Loop over the models
    for model_name in MODELS.keys():
        cfg = configure_model(model_name)  # configure the model
        metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0]
        )  # get metadata of the dataset
        output_file = os.path.join(
            output_dir, f"{model_name.replace('/', '_')}_results.csv"
        )  # path to the output CSV file

        # Write the CSV header
        with open(output_file, "w", newline="") as csvfile:
            filewriter = csv.writer(csvfile, delimiter=",")
            filewriter.writerow(
                ["Image_Path", "Classes"]
            )  # write header to the CSV file

        # Loop over every image in the directory
        print("Start detection for model: ", model_name)
        for root, dirs, files in os.walk(image_dir):
            print(f"Processing images in directory: {root}")
            for filename in tqdm(files):  # use tqdm to display progress
                if filename.endswith(".jpg") or filename.endswith(
                    ".png"
                ):  # process only jpg and png images
                    image_path = os.path.join(root, filename)
                    outputs = run_detection(cfg, image_path)  # perform object detection
                    write_results_to_csv(
                        output_file, image_path, outputs, metadata
                    )  # write results to the CSV file

        print("Detection finished for model: ", model_name)
        print("Results saved to: ", output_file)
