from eval_corpus import search_corpus_inconsistency, find_duplicates
from helper_functions import (
    find_objects,
    get_unique_values,
    get_unique_values_and_count_images,
    load_and_clean_data,
)
from create_format_dataset import (
    assign_labels as format_assign_labels,
    organize_images as format_organize_images,
)
from create_keyword_dataset import (
    assign_labels as keyword_assign_labels,
    organize_images as keyword_organize_images,
)

from create_vismig_dataset import create_vismig_dataset

# Define directories for CSV files and image files
csv_dir = "./corpus/csv"
img_dir = "./corpus/img"

# CSV files
csv_files = ["vif.csv", "baci.csv", "siba.csv", "polos.csv", "vismig.csv"]

# Image columns in each csv
image_columns = {
    "vif.csv": ["Image " + str(i) for i in range(1, 20)],
    "baci.csv": ["Image 1"],
    "siba.csv": ["Image 1"],
    "polos.csv": ["Image 1", "Image 2"],
    "vismig.csv": ["Image " + str(i) for i in range(1, 19)],
}

# List of problematic entries
problematic_entries = {
    "vismig.csv": [
        "o:vismig.m.2017.1",
        "o:vismig.m.2017.2",
        "o:vismig.m.2017.3",
        "o:vismig.m.2017.4",
        "o:vismig.m.2017.5",
        "o:vismig.m.2017.6",
    ]
}

# List of duplicates
duplicate_img_to_remove = {
    "vif.csv": [
        "vase.2782-IMAGE.2.jpg",
        "vase.2772-IMAGE.2.jpg",
        "vase.2777-IMAGE.2.jpg",
        "vase.2337-IMAGE.2.jpg",
        "vase.2781-IMAGE.2.jpg",
        "vase.2336-IMAGE.2.jpg",
        "vase.1042-IMAGE.1.jpg",
        "vase.2776-IMAGE.2.jpg",
        "vase.2769-IMAGE.2.jpg",
        "vase.2774-IMAGE.2.jpg",
        "vase.1044-IMAGE.2.jpg",
        "vase.2775-IMAGE.2.jpg",
        "vase.2773-IMAGE.2.jpg",
        "vase.2783-IMAGE.2.jpg",
        "vase.2768-IMAGE.2.jpg",
        "vase.2335-IMAGE.2.jpg",
        "vase.2778-IMAGE.2.jpg",
        "vase.2784-IMAGE.2.jpg",
        "vase.2785-IMAGE.2.jpg",
        "vase.2771-IMAGE.2.jpg",
        "vase.2770-IMAGE.2.jpg",
        "vase.2779-IMAGE.2.jpg",
        "vase.2780-IMAGE.2.jpg",
        "vase.2312-IMAGE.1.jpg",
        "vase.216-IMAGE.2.jpg",
        "vase.216-IMAGE.1.jpg",
        "vase.2296-IMAGE.1.jpg",
    ],
    "polos.csv": ["polos_1284_r.jpg", "polos_1208_r.jpg"],
}


format_labels = {
    "Photographs": [
        "Photographic plate",
        "Still photograph",
        "Photograph (copy)",
        "Photograph",
        "Photo reproduction",
        "Photographic negative",
        "Photographs",
        "Reproduction",
        "Carte de visite",
        "Diapositive",
    ],
    "Postcards": [
        "Photo postcard",
        "Postcard",
        "Real photo postcard",
        "Correspondence card",
    ],
    "Print media": [
        "Newspaper announcement",
        "Newspaper advertisement",
        "Newspaper",
        "Newspaper reports",
        "Magazine",
    ],
    "Visual art": ["Drawing", "Collage", "Painting", "Sign"],
    "Print materials": ["Poster", "Folder", "Leaflet"],
}

if __name__ == "__main__":
    # eval corpus
    # eval = search_corpus_inconsistency(image_columns, csv_dir, img_dir)

    # find duplicates
    # duplicates = find_duplicates(img_dir, csv_files)
    # for duplicate in duplicates:
    #     print(f"Duplicate found: {duplicate[0]} and {duplicate[1]}")

    # load and clean data (to test)
    # for file_name in csv_files:
    #     df = load_and_clean_data(csv_dir, file_name, problematic_entries, duplicate_img_to_remove)

    # Get unique values for a specific column
    # column_name = 'Type'
    # unique_values = get_unique_values(column_name, csv_dir, csv_files, problematic_entries, duplicate_img_to_remove)
    # print(unique_values)

    # Find objects with a specific value in a specific column
    # column_name = 'Type'
    # column_value = 'Diapositive'
    # matching_pids = find_objects(column_name, column_value, csv_dir, csv_files, problematic_entries, duplicate_img_to_remove)
    # print(matching_pids)

    # Get unique values and their object and image counts for a specific column
    # column_name = 'Type'
    # unique_values_and_counts = get_unique_values_and_count_images(column_name, csv_dir, csv_files, image_columns, problematic_entries, duplicate_img_to_remove)
    # print(unique_values_and_counts)

    # Assign labels to images for the format classification task
    # format_label_data = format_assign_labels(format_labels, csv_dir, csv_files, problematic_entries, image_columns, duplicate_img_to_remove)

    # Organize images into folders based on their labels
    # format_organize_images(format_label_data, img_dir)

    # Assign labels to images for the keyword classification task
    # keyword = '714'
    # column = 'OCM'
    # collections_to_exclude = ['polos.csv', 'vismig.csv', 'baci.csv']
    # keyword_label_data = keyword_assign_labels(keyword, column, csv_dir, csv_files, problematic_entries, image_columns, duplicate_img_to_remove, collections_to_exclude=collections_to_exclude)
    # keyword_organize_images(keyword_label_data, img_dir)

    # Assign labels to images for the keyword classification task
    keyword = "500"
    column = "OCM"
    collections_to_exclude = ["vismig.csv", "baci.csv", "polos.csv"]
    keyword_label_data = keyword_assign_labels(
        keyword,
        column,
        csv_dir,
        csv_files,
        problematic_entries,
        image_columns,
        duplicate_img_to_remove,
        collections_to_exclude=collections_to_exclude,
    )
    keyword_organize_images(keyword_label_data, img_dir)

    # Assign labels to images for the keyword classification task
    # keyword = 'S025' #S002
    # column = 'Internal keywords'
    # collections_to_exclude = ['vif', 'siba.csv', 'baci.csv', 'polos.csv']
    # keyword_label_data = keyword_assign_labels(keyword, column, csv_dir, csv_files, problematic_entries, image_columns, duplicate_img_to_remove, collections_to_exclude=collections_to_exclude)
    # keyword_organize_images(keyword_label_data, img_dir)

    # create vismig dataset
    # create_vismig_dataset(csv_dir, problematic_entries, duplicate_img_to_remove)
