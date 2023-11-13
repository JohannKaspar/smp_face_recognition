from collections import defaultdict
from pathlib import Path

import numpy as np

DATASET_ROOT = Path("Dataset")

IMAGES_ROOT = DATASET_ROOT / "originalPics"

EXTRACTED_IMAGES_ROOT = DATASET_ROOT / "extractedPics"

FOLDS_ROOT = DATASET_ROOT / "FDDB-folds"

# The folds of the dataset
FOLDS = {
    "TRAIN" : [f"FDDB-fold-0{i}-ellipseList.txt" for i in range(1, 7)],
    "VAL": [f"FDDB-fold-0{i}-ellipseList.txt" for i in range(7, 9)],
    "TEST": ["FDDB-fold-09-ellipseList.txt", "FDDB-fold-10-ellipseList.txt"]
}


def read_fold_content(fold_root, fold_name):
    """
    Reads the content of a fold file and returns it as a list of strings.

    Args:
        fold_root (Path): The root directory of the fold file.
        fold_name (str): The name of the fold file.

    Returns:
        List[str]: The content of the fold file as a list of strings.
    """
    # Read the content of the fold file as a string and split it into a list of strings
    return (fold_root / fold_name).read_text().split("\n")


from collections import defaultdict

def parse_fold_annotations(fold_root: str, fold_name: str) -> tuple:
    """
    Parses the annotations for a given fold of the SMP dataset.

    Args:
        fold_root (str): The root directory of the SMP dataset.
        fold_name (str): The name of the fold to parse annotations for.

    Returns:
        A tuple containing two dictionaries:
        - annotations_per_image: A dictionary mapping image paths to a list of annotations.
        - num_annotations_per_image: A dictionary mapping image paths to the number of annotations for that image.
    """
    # Initialize two dictionaries to store the annotations and the number of annotations per image
    annotations_per_image = defaultdict(list)
    num_annotations_per_image = defaultdict(int)

    # Initialize the current path variable
    current_path = ""

    # Read the content of the fold
    folds_content = read_fold_content(fold_root, fold_name)

    # Loop through the lines of the fold content
    for line in folds_content:
        if line == "":
            continue

        # If the line starts with "2002/" or "2003/", it is a new image path
        elif line.startswith("2002/") or line.startswith("2003/"):
            current_path = line

            # Initialize an empty list for the annotations of the current image
            annotations_per_image[current_path] = []
            
        # If the line is a digit, it is the number of annotations for the current image
        elif line.isdigit():
            num_annotations_per_image[current_path] = int(line)

        # Otherwise, it is an annotation for the current image
        else:
            annotations_per_image[current_path].append(line)

    # Return the two dictionaries as a tuple
    return annotations_per_image, num_annotations_per_image


from collections import defaultdict
import numpy as np

def read_annotations(folds_root, folds):
    """
    Reads annotations for each image in the specified folds.

    Args:
        folds_root (str): Root directory of the folds.
        folds (list): List of fold numbers to read annotations for.

    Returns:
        dict: A dictionary where the keys are image paths and the values are lists of bounding boxes for that image.
              Each bounding box is represented as a tuple of (h_min, w_min, h_max, w_max).
    """
    all_annotations_per_image = defaultdict(list)

    for fold in folds:
        # Parse annotations for the current fold
        annotations_per_image_raw, num_annotations_per_image = parse_fold_annotations(folds_root, fold)

        for image_path, image_annotations_raw in annotations_per_image_raw.items():

            # Consistency check
            assert len(image_annotations_raw) == num_annotations_per_image[image_path]

            for annotation in image_annotations_raw:
                # Parse annotation parameters
                parameters = [float(value) for value in annotation.split(" ")[:-2]]

                # Extract parameters
                major_axis_radius, minor_axis_radius, angle, center_w, center_h = parameters

                # Compute bounding box half-widths
                bbox_w_half_width = np.hypot(major_axis_radius * np.cos(angle), minor_axis_radius * np.sin(angle))
                bbox_h_half_width = np.hypot(major_axis_radius * np.sin(angle), minor_axis_radius * np.cos(angle))

                # Compute bounding box coordinates
                w_min = center_w - bbox_w_half_width
                h_min = center_h - bbox_h_half_width

                w_max = center_w + bbox_w_half_width
                h_max = center_h + bbox_h_half_width

                # Add bounding box to list of annotations for the current image
                all_annotations_per_image[image_path].append((h_min, w_min, h_max, w_max))

    return all_annotations_per_image
