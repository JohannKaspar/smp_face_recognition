
import numpy as np
import skimage
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle


def load_image_to_array(image_path):
    img = skimage.io.imread(image_path)

    # Check for grayscale images
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    return np.asarray(img)


def plot_image_with_bounding_boxes(img_array, annotations_for_image, predictions_for_image = ()):
    plt.imshow(img_array)
    ax = plt.gca()

    for gt_bbox in annotations_for_image:
        ax.add_patch(
            Rectangle((gt_bbox[1], gt_bbox[0]), gt_bbox[3]-gt_bbox[1], gt_bbox[2]-gt_bbox[0], fill=None, linewidth=3, color="r")
        )

    for pred_bbox in predictions_for_image:
        ax.add_patch(
            Rectangle((pred_bbox[1], pred_bbox[0]), pred_bbox[3]-pred_bbox[1], pred_bbox[2]-pred_bbox[0], fill=None, linewidth=3, color="b")
        )

    plt.show()
