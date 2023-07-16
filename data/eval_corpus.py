import os
import pandas as pd
import numpy as np
import hashlib

# The following function search_corpus_inconsistency checks the consistency of a dataset by searching for missing or duplicated images.
def search_corpus_inconsistency(image_columns, csv_dir, img_dir):
    """
    Search for inconsistencies (missing or duplicated images) in a dataset (based on img names).

    Args:
    image_columns (dict): A dictionary mapping CSV files to their corresponding image columns.
    csv_dir (str): The directory where the CSV files are stored.
    img_dir (str): The directory where the image files are stored.

    Returns:
    None
    """
    # Initialize a set to track all processed images
    seen_images = set()

    # Iterate over each CSV file
    for csv_file, img_columns in image_columns.items():
        # Load the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(csv_dir, csv_file))

        # Display column names for debugging purposes
        print(df.columns)

        # Handle inconsistencies in column names by removing any leading or trailing whitespaces, and unnecessary double quotes
        df.columns = df.columns.str.replace('"', '').str.strip()

        # Remove leading and trailing whitespaces in each column's values, and replace empty strings with NaN
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x).replace(r'^\s*$', np.nan, regex=True)

        # Iterate over each specified image column in the current CSV file
        for column in img_columns:
            # Ensure the specified column exists in the DataFrame
            if column in df.columns:
                # Drop any rows in the column where the value is NaN
                valid_images = df[column].dropna()

                # Iterate over each image in the current column
                for image in valid_images:
                    # Construct the path to the image file
                    img_path = os.path.join(img_dir, csv_file.replace('.csv', ''), image.strip())

                    # Print a warning if the image file does not exist, or if it has been processed before
                    if not os.path.isfile(img_path):
                        print(f"Warning: Missing image at {img_path}")
                    else:
                        if img_path in seen_images:
                            print(f"Warning: Repeated image at {img_path}")
                        else:
                            seen_images.add(img_path)
            else:
                print(f"Warning: Column {column} not found in {csv_file}")

# Note: There are issues with the objects vismig.m.2017.1 - vismig.m.2017.6 in the 'vismig' collection. These objects will be removed from the dataframe in subsequent processing. See helper_functions.py --> remove_problematic_entries()
# Note: column names and values are to clean! See helper_functions.py --> clean_data()


# The following function find_duplicates identifies duplicated images by computing a unique hash value for each image file.
def find_duplicates(img_dir, csv_files):
    """
    Find duplicated images in a dataset by computing a unique hash value for each image file.

    Args:
    img_dir (str): The directory where the image files are stored.
    csv_files (list): List of CSV file names.

    Returns:
    list: A list of tuples. Each tuple contains two identical images.
    """
    # Initialize a dictionary to store the MD5 hash of each image file and the file's path.
    hash_dict = {}
    # Initialize a list to store the paths of duplicated images.
    duplicates = []

    # Iterate over each CSV file name.
    for file_name in csv_files:
        # Get the corresponding image directory name by removing the '.csv' extension from the CSV file name.
        img_folder_path = os.path.join(img_dir, file_name.replace('.csv', ''))

        # Iterate over each image file in the current image directory.
        for image in os.listdir(img_folder_path):
            # Process only images.
            if image.endswith('.jpg') or image.endswith('.jpeg'):
                # Get the full path of the current image file.
                img_path = os.path.join(img_folder_path, image)

                # Initialize an MD5 hash object.
                file_hash = hashlib.md5()
                # Open the current image file in binary mode.
                with open(img_path, 'rb') as f:
                    # Read the file in chunks of 8192 bytes and update the hash object.
                    while chunk := f.read(8192):
                        file_hash.update(chunk)
                # Get the hexadecimal representation of the hash value.
                file_hash = file_hash.hexdigest()

                # If the hash value is already in the dictionary, it means the current image is a duplicate.
                if file_hash in hash_dict:
                    # Add a tuple containing the paths of the duplicate images to the list.
                    duplicates.append((hash_dict[file_hash], img_path))
                else:
                    # If the hash value is not in the dictionary, add it and associate it with the current image path.
                    hash_dict[file_hash] = img_path

    # Return the list of duplicates.
    return duplicates
