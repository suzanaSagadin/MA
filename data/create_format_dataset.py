import pandas as pd
import os
import shutil
from helper_functions import load_and_clean_data

# This function is used to assign labels to each image in our dataset. 
def assign_labels(format_labels, csv_dir, csv_files, problematic_entries, image_columns, duplicate_img_to_remove):
    """
    Function to assign labels to each image in a dataset based on certain characteristics. 

    Args:
    format_labels (dict): A dictionary to be reversed for label lookups.
    csv_dir (str): The directory where CSV files are located.
    csv_files (list): A list of CSV files to process.
    problematic_entries (list): A list of entries that cause problems during processing.
    image_columns (dict): A dictionary specifying which columns to use in each CSV file.
    duplicate_img_to_remove (list): A list of duplicate images to remove from processing.

    Returns:
    label_data (DataFrame): A DataFrame containing the collection, image name, and assigned label for each image.
    """

    # Reversing the given dictionary so we can look up the key by its values.
    reverse_format_labels = {v: k for k, values in format_labels.items() for v in values}

    # Defining which image names indicate it's a 'recto' (front side of a leaf)
    recto_strings = ['_r.', '-IMAGE.2', '-IMAGE.3', '-IMAGE.4', '-IMAGE.5', '-IMAGE.6', '-IMAGE.7', '-IMAGE.8', '-IMAGE.9', '-IMAGE.10', '-IMAGE.11', '-IMAGE.12', '-IMAGE.13', '-IMAGE.14', '-IMAGE.15', '-IMAGE.16', '-IMAGE.17', '-IMAGE.18', '-IMAGE.19']

    # Some exceptions that are not 'recto' despite the above naming conventions.
    not_recto = ['o:vase.2049', 'o:vase.2050', 'o:vase.2070', 'o:vase.2075', 'o:vase.2075', 'o:vase.2075', 'o:vase.2146', 'o:vase.2146', 'o:vase.2200', 'o:vase.2276', 'o:vase.2397', 'o:vase.2144']

    # Initializing an empty dataframe that will store our data for labeling.
    label_data = pd.DataFrame(columns=['collection', 'img', 'label'])

    # Looping through each file in our list of CSV files.
    for file_name in csv_files:
        # Load and clean data
        df = load_and_clean_data(csv_dir, file_name, problematic_entries, duplicate_img_to_remove)
        print(f'Loaded and cleaned {file_name}')

        # Looping through each row in our data.
        for index, row in df.iterrows():
            # Looping through each column defined in image_columns for the current file.
            for column in image_columns[file_name]:
                # Checking if the cell has a value.
                if pd.notnull(row[column]):
                    # Looking up the label for this type or assigning 'Unknown' if not found.
                    label = reverse_format_labels.get(row['Type'], 'Unknown')

                    # If the label is 'Photographs' or 'Postcards', we determine if it's a recto or verso.
                    if label in ['Photographs', 'Postcards']:
                        # Check if the item is a recto and is not in the exception list.
                        if any(substr in row[column] for substr in recto_strings) and row['PID'] not in not_recto:
                            label = label + ' recto'
                        else:
                            label = label + ' verso'

                    # Adding this row to our label_data dataframe.
                    new_data = pd.DataFrame({'collection': [file_name.replace('.csv', '')], 'img': [row[column]], 'label': [label]})
                    label_data = pd.concat([label_data, new_data], ignore_index=True)
    
    # Save the completed label_data to a csv file.
    label_data.to_csv('datasets/format_label_data.csv', index=False)
    return label_data

# This function is used to organize the images into directories based on their labels.
def organize_images(label_data, img_dir):
    """
    Function to organize image files into directories based on their assigned labels. 

    Args:
    label_data (DataFrame): A DataFrame containing the collection, image name, and assigned label for each image.
    img_dir (str): The directory where the unorganized images are located.

    Returns:
    None
    """

    # Setting the base directory for our organized images.
    base_dir = 'datasets/format'

    # If the base directory doesn't exist, we create it.
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Looping through each row in our label data.
    for index, row in label_data.iterrows():
        # Setting the source directory (where our unorganized images are).
        src_dir = os.path.join(img_dir, row['collection'])

        # Setting the destination directory (where we want the organized images to go).
        dst_dir = os.path.join(base_dir, row['label'])

        # If the destination directory doesn't exist, we create it.
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        # Defining the source and destination file paths.
        src_file = os.path.join(src_dir, row['img'])
        dst_file = os.path.join(dst_dir, row['img'])

        # Checking if the source file exists before trying to copy it.
        if not os.path.isfile(src_file):
            print(f"Warning: source file not found {src_file}")
            continue

        # Copying the file from the source to the destination directory.
        shutil.copy(src_file, dst_file)

    print("Image organization complete!")
