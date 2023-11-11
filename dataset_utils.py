from collections import defaultdict
from pathlib import Path

import numpy as np

DATASET_ROOT = Path("Dataset")

IMAGES_ROOT = DATASET_ROOT / "originalPics"

EXTRACTED_IMAGES_ROOT = DATASET_ROOT / "extractedPics"

FOLDS_ROOT = DATASET_ROOT / "FDDB-folds"

FOLDS = {
    "TRAIN" : [f"FDDB-fold-0{i}-ellipseList.txt" for i in range(1, 7)],
    "VAL": [f"FDDB-fold-0{i}-ellipseList.txt" for i in range(7, 9)],
    "TEST": ["FDDB-fold-09-ellipseList.txt", "FDDB-fold-10-ellipseList.txt"]
}


def read_fold_content(fold_root, fold_name):
    return (fold_root / fold_name).read_text().split("\n")


def parse_fold_annotations(fold_root, fold_name):
    annotations_per_image = defaultdict(list)
    num_annotations_per_image = defaultdict(int)

    current_path = ""

    folds_content = read_fold_content(fold_root, fold_name)

    for line in folds_content:
        if line == "":
            continue

        elif line.startswith("2002/") or line.startswith("2003/"):
            current_path = line

            annotations_per_image[current_path] = []
            
        elif line.isdigit():
            num_annotations_per_image[current_path] = int(line)

        else:
            annotations_per_image[current_path].append(line)

    return annotations_per_image, num_annotations_per_image


def read_annotations(folds_root, folds):

    all_annotations_per_image = defaultdict(list)

    for fold in folds:
        annotations_per_image_raw, num_annotations_per_image = parse_fold_annotations(folds_root, fold)

        for image_path, image_annotations_raw in annotations_per_image_raw.items():

            # Consistency check
            assert len(image_annotations_raw) == num_annotations_per_image[image_path]

            for annotation in image_annotations_raw:
                parameters = [float(value) for value in annotation.split(" ")[:-2]]

                major_axis_radius, minor_axis_radius, angle, center_w, center_h = parameters

                bbox_w_half_width = np.hypot(major_axis_radius * np.cos(angle), minor_axis_radius * np.sin(angle))
                bbox_h_half_width = np.hypot(major_axis_radius * np.sin(angle), minor_axis_radius * np.cos(angle))

                w_min = center_w - bbox_w_half_width
                h_min = center_h - bbox_h_half_width

                w_max = center_w + bbox_w_half_width
                h_max = center_h + bbox_h_half_width

                all_annotations_per_image[image_path].append((h_min, w_min, h_max, w_max))

    return all_annotations_per_image
