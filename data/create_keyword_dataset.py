import pandas as pd
import shutil
import os
from helper_functions import load_and_clean_data

def assign_labels(keyword, column, csv_dir, csv_files, problematic_entries, image_columns, duplicate_img_to_remove, collections_to_exclude=None):
    """
    Assign labels to the images in each dataset based on the presence of a keyword.

    Args:
    keyword (str): The keyword to check for in the data.
    column (str): The column in the data to check for the keyword.
    csv_dir (str): The directory containing the CSV files.
    csv_files (list): The names of the CSV files to process.
    problematic_entries (list): Entries that should be removed during processing.
    image_columns (dict): The columns in each CSV file that contain image data.
    duplicate_img_to_remove (list): List of duplicate images to remove.
    collections_to_exclude (list): Collections to exclude from processing, default is None.

    Returns:
    label_data (pd.DataFrame): DataFrame containing collection, image and assigned label.
    """
    
    # specific strings indicating the recto side of images
    recto_strings = ['_r.', '-IMAGE.2', '-IMAGE.3', '-IMAGE.4', '-IMAGE.5', '-IMAGE.6', '-IMAGE.7', '-IMAGE.8', '-IMAGE.9', '-IMAGE.10', '-IMAGE.11', '-IMAGE.12', '-IMAGE.13', '-IMAGE.14', '-IMAGE.15', '-IMAGE.16', '-IMAGE.17', '-IMAGE.18', '-IMAGE.19']

    # specific objects that are not recto after all
    not_recto = ['o:vase.2049', 'o:vase.2050', 'o:vase.2070', 'o:vase.2075', 'o:vase.2075', 'o:vase.2075', 'o:vase.2146', 'o:vase.2146', 'o:vase.2200', 'o:vase.2276', 'o:vase.2397', 'o:vase.2144']

    # create an empty DataFrame to store the labels
    label_data = pd.DataFrame(columns=['collection', 'img', 'label'])

    # if collections_to_exclude is not provided, set it to an empty list
    if collections_to_exclude is None:
        collections_to_exclude = []

    # iterating over all csv files
    for file_name in csv_files:
        # skip files that are in the collections_to_exclude list
        if file_name in collections_to_exclude:
            continue

        # load and clean the data
        df = load_and_clean_data(csv_dir, file_name, problematic_entries, duplicate_img_to_remove)
        print(f'Loaded and cleaned {file_name}')

        # iterating over all rows of the DataFrame
        for index, row in df.iterrows():
            # check if the keyword is present in the specified column
            contains_keyword = keyword in str(row[column]).split("; ")

            # iterating over all valid image columns
            for img_column in image_columns[file_name]:
                # if the cell contains image data
                if pd.notnull(row[img_column]):
                    # check if it's not a recto image or it's in the not_recto list, then add it to label_data
                    if not any(substr in row[img_column] for substr in recto_strings) or row['PID'] in not_recto:
                        # determine the label
                        label = keyword if contains_keyword else 'other'

                        # add a new row to label_data
                        new_data = pd.DataFrame({'collection': [file_name.replace('.csv', '')], 'img': [row[img_column]], 'label': [label]})
                        label_data = pd.concat([label_data, new_data], ignore_index=True)

    # save label_data to a csv file
    label_data.to_csv('datasets/keyword_' + keyword + '_label_data.csv', index=False)
    return label_data


def organize_images(label_data, img_dir):
    """
    Organize the images based on the labels assigned.

    Args:
    label_data (pd.DataFrame): DataFrame containing the image labels.
    img_dir (str): The directory containing the images.

    Returns:
    None
    """
    # define the base directory
    base_dir = 'datasets/keywords'

    # extract the keyword from the label data that is not 'other'
    keyword = label_data[label_data['label'] != 'other']['label'].iloc[0]

    # iterating over all rows in the label_data DataFrame
    for index, row in label_data.iterrows():
        # define the source path for the image
        source_path = os.path.join(img_dir, row['collection'], row['img'])
        if not os.path.isfile(source_path):
            print(f"File not found: {source_path}")
            continue

        # determine the name of the subfolder
        subfolder = row['label'] if row['label'] != 'other' else 'other'

        # define the destination path for the image
        keyword_folder = os.path.join(base_dir, str(keyword))
        dest_folder = os.path.join(keyword_folder, subfolder)
        os.makedirs(dest_folder, exist_ok=True)
        dest_path = os.path.join(dest_folder, row['img'])

        # copy the image from the source to the destination
        shutil.copyfile(source_path, dest_path)

    print(f"Organized images for keyword '{keyword}' in '{base_dir}/{keyword}'")

