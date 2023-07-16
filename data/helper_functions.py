import pandas as pd
import numpy as np
import os

def remove_problematic_entries(df, problematic_entries):
    """
    Remove rows from a DataFrame that contain problematic entries.

    Args:
    df (pandas.DataFrame): The input DataFrame.
    problematic_entries (list): A list of problematic entry identifiers.

    Returns:
    pandas.DataFrame: The cleaned DataFrame.
    """
    # Iterate over all problematic entries
    for entry in problematic_entries:
        # Find rows where the PID is in the list of problematic entries
        problematic_rows = df[df['PID'] == entry]
        # If any problematic rows are found, print a message for each
        if not problematic_rows.empty:  
            for pid in problematic_rows['PID']:
                print(f"Row with PID {pid} removed.")
        # Only keep the rows where the PID is not in the list of problematic entries
        df = df[df['PID'] != entry]
    return df

def remove_duplicate_images(df, duplicate_img_to_remove):
    """
    Remove duplicate images from a DataFrame.

    Args:
    df (pandas.DataFrame): The input DataFrame.
    duplicate_img_to_remove (list): A list of duplicate images to remove.

    Returns:
    pandas.DataFrame: The DataFrame with duplicate images removed.
    """
    # Iterate over all duplicate images
    for image in duplicate_img_to_remove:
        # Iterate over all columns in the dataframe that start with 'Image'  
        for column in df.columns:
            if column.startswith('Image'):
                # If the image is found in the column, replace the cell value with None
                if df[column].eq(image).any():
                    df.loc[df[column] == image, column] = np.nan
                    print(f"Duplicate image {image} removed from column {column}.")
                    break
    return df

def clean_data(df, file_name, problematic_entries, duplicate_img_to_remove):
    """
    Clean the DataFrame by cleaning column names, values, and removing problematic entries and duplicate images.

    Args:
    df (pandas.DataFrame): The input DataFrame.
    file_name (str): The name of the csv file.
    problematic_entries (dict): A dictionary of problematic entries to remove.
    duplicate_img_to_remove (dict): A dictionary of duplicate images to remove.

    Returns:
    pandas.DataFrame: The cleaned DataFrame.
    """
    # Clean column names
    df.columns = df.columns.str.replace('"', '').str.strip()

    # Clean column values
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x).replace(r'^\s*$', np.nan, regex=True)

    # If there are any problematic entries for this csv file, remove them
    if file_name in problematic_entries:
        df = remove_problematic_entries(df, problematic_entries[file_name])

    # If there are any duplicate images for this csv file, remove them
    if file_name in duplicate_img_to_remove:
        df = remove_duplicate_images(df, duplicate_img_to_remove[file_name])

    return df

def load_and_clean_data(csv_dir, file_name, problematic_entries, duplicate_img_to_remove):
    """
    Load a csv file into a DataFrame and clean the data.

    Args:
    csv_dir (str): The directory containing the csv files.
    file_name (str): The name of the csv file to load.
    problematic_entries (dict): A dictionary of problematic entries to remove.
    duplicate_img_to_remove (dict): A dictionary of duplicate images to remove.

    Returns:
    pandas.DataFrame: The cleaned DataFrame.
    """
    # File path
    file_path = os.path.join(csv_dir, file_name)

    # Read csv
    df = pd.read_csv(file_path)

    # Clean data
    df = clean_data(df, file_name, problematic_entries, duplicate_img_to_remove)

    return df

def get_unique_values(column_name, csv_dir, csv_files, problematic_entries, duplicate_img_to_remove):
    """
    Get all unique values in a column across multiple csv files.

    Args:
    column_name (str): The name of the column.
    csv_dir (str): The directory containing the csv files.
    csv_files (list): A list of csv file names.
    problematic_entries (dict): A dictionary of problematic entries to remove.
    duplicate_img_to_remove (dict): A dictionary of duplicate images to remove.

    Returns:
    set: A set of all unique values in the column.
    """
    # All unique values in the given column
    all_values = set()

    for file_name in csv_files:
        # Load and clean data
        df = load_and_clean_data(csv_dir, file_name, problematic_entries, duplicate_img_to_remove)

        # Check if the given column exists after cleaning
        if column_name in df.columns:
            # Get unique values
            values = set(df[column_name].dropna().unique())

            # Add to all values
            all_values.update(values)
        else:
            print(f"Warning: '{column_name}' column not found in {file_name}.")

    return all_values

def get_unique_values_and_count_images(column_name, csv_dir, csv_files, image_columns, problematic_entries, duplicate_img_to_remove):
    """
    Get all unique values in a column and count the associated images.

    Args:
    column_name (str): The name of the column.
    csv_dir (str): The directory containing the csv files.
    csv_files (list): A list of csv file names.
    image_columns (dict): A dictionary of image column names.
    problematic_entries (dict): A dictionary of problematic entries to remove.
    duplicate_img_to_remove (dict): A dictionary of duplicate images to remove.

    Returns:
    dict: A dictionary of unique values and their counts.
    """
    # Dictionary of all unique values and their counts
    value_counts = {}

    for file_name in csv_files:
        # Load and clean data
        df = load_and_clean_data(csv_dir, file_name, problematic_entries, duplicate_img_to_remove)

        # Check if the given column exists after cleaning
        if column_name in df.columns:
            # Iterate over unique values in column
            for value in df[column_name].unique():
                # If the value is not null
                if pd.notnull(value):
                    # Select rows where column equals value
                    sub_df = df[df[column_name] == value]

                    # Initialize image count for this value
                    image_count = 0

                    # If the file_name is in image_columns, we check each image column
                    if file_name in image_columns:
                        for img_column in image_columns[file_name]:
                            # If the image column is in the sub-dataframe, add up the non-null values
                            if img_column in sub_df.columns:
                                image_count += sub_df[img_column].notna().sum()

                    # If the value is not in value_counts, initialize a dictionary for it
                    if value not in value_counts:
                        value_counts[value] = {'object_count': 0, 'image_count': 0}

                    # Increment the counts
                    value_counts[value]['object_count'] += len(sub_df)
                    value_counts[value]['image_count'] += image_count
        else:
            print(f"Warning: '{column_name}' column not found in {file_name}.")

    return value_counts

def find_objects(column_name, column_value, csv_dir, csv_files, problematic_entries, duplicate_img_to_remove):
    """
    Find all records in the csv files that match a given column value.

    Args:
    column_name (str): The name of the column.
    column_value (str): The value to search for.
    csv_dir (str): The directory containing the csv files.
    csv_files (list): A list of csv file names.
    problematic_entries (dict): A dictionary of problematic entries to remove.
    duplicate_img_to_remove (dict): A dictionary of duplicate images to remove.

    Returns:
    list: A list of all matching PIDs.
    """
    # List of all matching PIDs
    all_pids = []

    for file_name in csv_files:
        # Load and clean data
        df = load_and_clean_data(csv_dir, file_name, problematic_entries, duplicate_img_to_remove)

        # Check if the given column exists after cleaning
        if column_name in df.columns:
            # Get the PIDs of the matching records
            matching_pids = df.loc[df[column_name] == column_value, 'PID'].tolist()

            # Add to the list of all PIDs
            all_pids.extend(matching_pids)
        else:
            print(f"Warning: '{column_name}' column not found in {file_name}.")

    return all_pids
