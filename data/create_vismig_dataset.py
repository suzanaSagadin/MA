import os
import shutil
from helper_functions import load_and_clean_data


# function create_vismig_dataset
def create_vismig_dataset(csv_dir, problematic_entries, duplicate_img_to_remove):
    file_name = "vismig.csv"
    df = load_and_clean_data(
        csv_dir, file_name, problematic_entries, duplicate_img_to_remove
    )
    base_dir = "datasets/vismig"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # get all the img_names from df; they are in the column 'Image 1' to 'Image 18'
    img_names = []
    for i in range(1, 19):
        img_names += df["Image " + str(i)].tolist()
    # remove nan values
    img_names = [x for x in img_names if str(x) != "nan"]
    # source path for each image; 'corpus/img/vismig' + img_name
    source_paths = [os.path.join("corpus/img/vismig", x) for x in img_names]
    # target path for each image; 'datasets/vismig' + img_name
    target_paths = [os.path.join(base_dir, x) for x in img_names]
    # copy images from source to target
    for source, target in zip(source_paths, target_paths):
        shutil.copy(source, target)
