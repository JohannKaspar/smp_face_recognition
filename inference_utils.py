import keras_cv
import numpy as np
import tensorflow as tf
from PIL import Image

EMPTY_LABELS = {
    "boxes": tf.ragged.constant(
        [
            [[-1., -1., -1., -1.]],
        ],
    ragged_rank=1,
    dtype=tf.float32,
),
    "classes": tf.ragged.constant(
        [
            [-1.],
        ],
    ragged_rank=1,
    dtype=tf.float32,
)
}
EMPTY_DETECTIONS = {
    "boxes": tf.ragged.constant(
        [
            [[-1., -1., -1., -1.]],
        ],
    ragged_rank=1,
    dtype=tf.float32,
    ),
    "classes": tf.ragged.constant(
        [
            [-1.],
        ],
    ragged_rank=1,
    dtype=tf.float32,
),
    "confidence": tf.ragged.constant(
        [
            [-1.],
        ],
    ragged_rank=1,
    dtype=tf.float32,
)
}


def get_sliding_window_corners(img_dim, dim):
    """
    Returns the starting coordinates of the sliding window for a given image dimension and window size.

    Args:
    img_dim: int, the dimension of the image
    dim: int, the size of the sliding window

    Returns:
    box_begins: list of ints, the starting coordinates of the sliding window
    """
    # Calculate the number of sliding windows that can fit in the image dimension
    n_tilde = img_dim // dim

    # Calculate the overlap between sliding windows
    overlap = int(((n_tilde + 1) * dim - img_dim) / n_tilde)  # beispiel: Bildbreite 120, Boxbreite 50, n_tilde = 2, overlap = 10

    # Calculate the starting coordinates of each sliding window
    box_begins = []
    for i in range(n_tilde + 1):  # beispiel: n_tilde = 2, i = 0, 1, 2
        bgn = i * (dim - overlap) # beispiel: i = 0, bgn = 0, i = 1, bgn = 40, i = 2, bgn = 80

        # If this is the last sliding window, adjust the starting coordinate to ensure it fits within the image
        if i == n_tilde:
            bgn = img_dim - dim
        box_begins.append(bgn)

    # add centers of boxes as additional box begins
    """for i in range(n_tilde):
        bgn = int(i * (dim - overlap) + dim / 2)
        box_begins.append(bgn)"""

    return box_begins


def get_sliding_window_inference_boxes(image_shape, input_shape):
    """
    Returns the coordinates of the sliding windows for a given image and input shape.

    Args:
    image_shape: tuple of ints, the shape of the image
    input_shape: tuple of ints, the shape of the input

    Returns:
    inference_boxes: list of tuples, the coordinates of the sliding windows
    """
    # Get the height, width and number of channels of the image
    image_height, image_width, _ = image_shape
    # Get the height and width of the input
    input_height, input_width = input_shape

    # Initialize an empty list to store the sliding windows
    inference_boxes = []

    # Loop over all possible sliding windows
    for h_min in get_sliding_window_corners(image_height, input_height):
        for w_min in get_sliding_window_corners(image_width, input_width):
            # Append the coordinates of the sliding window to the list
            inference_boxes.append(
                (h_min, w_min, h_min + input_height, w_min + input_width)
            )

    # Return the list of sliding windows
    return inference_boxes


def resize_image_pil(image, target_size):
    # Open the image using PIL
    img = Image.fromarray(image)

    # Resize the image to the target size
    img = img.resize(target_size)

    # Convert the image back to a NumPy array
    resized_image = np.array(img)

    return resized_image


def get_sliding_window_detections(img_array, input_shape, model, size_factors=[1]):
    """
    Returns the coordinates of the sliding windows where the model predicts an object.

    Args:
    img_array: numpy array, the image to be processed
    input_shape: tuple of ints, the shape of the input
    model: keras model, the model to be used for inference

    Returns:
    detection_boxes: list of tuples, the coordinates of the sliding windows where the model predicts an object
    """
    # Extract the sliding windows from the image
    inference_images = []

    inference_boxes = []

    for size_factor in size_factors:
        # Get the coordinates of the sliding windows
        this_size_inference_boxes = get_sliding_window_inference_boxes(
            img_array.shape, (input_shape[0] * size_factor, input_shape[1] * size_factor) 
        )  # (input_height, input_width)


        inference_boxes.extend(this_size_inference_boxes)
    
        for inference_box in inference_boxes:
            # resize image to fit to model.input.shape[1] and model.input.shape[2]

            image = img_array[
                    inference_box[0] : inference_box[2], inference_box[1] : inference_box[3]
                ]

            resized_image = resize_image_pil(image, (input_shape[1], input_shape[0]))
            
            inference_images.append(resized_image)

    # Stack the sliding windows into a batch
    batch = np.stack(inference_images)

    # Predict the class of each sliding window in the batch
    model_result = tf.math.argmax(
        tf.math.softmax(model.predict(batch), axis=-1), axis=-1
    ).numpy()
    # TODO: implement non-max surpression
    
    confidence_threshold = 0.5
    # get first array axis
    # delete all predictions where threshold is not met#
    model_result = np.delete(model_result, np.where(model_result[:, 1] < confidence_threshold), axis=0)

    # get box with highest threshold
    # calculate the iou of the box with the hiighest threshold and all other boxes
    # delete all boxes with iou > 0.5

    # Get the coordinates of the sliding windows where the model predicts an object
    detection_boxes = [
        inference_box
        for idx, inference_box in enumerate(inference_boxes)
        if model_result[idx]
    ]

    return detection_boxes


def get_sliding_window_detections_with_scores(
        img_array, 
        input_shape, 
        model, 
        confidence_threshold=0.5
    ):
    """
    Returns the coordinates of the sliding windows where the model predicts an object with confidence above a threshold.

    Args:
    img_array: numpy array, the image to be processed
    input_shape: tuple of ints, the shape of the input
    model: keras model, the model to be used for inference
    confidence_threshold: float, the confidence threshold for object detection

    Returns:
    detection_result: dictionary, the coordinates, classes, and confidence scores of the detected objects
    """
    # This function assumes that there is only one class besides the background!

    # Get the sliding window inference boxes
    inference_boxes = get_sliding_window_inference_boxes(
        img_array.shape, input_shape
    )  # (input_height, input_width)

    # Extract the inference images from the input image
    inference_images = []
    for inference_box in inference_boxes:
        inference_images.append(
            img_array[
                inference_box[0] : inference_box[2], inference_box[1] : inference_box[3]
            ]
        )

    # Stack the inference images into a batch
    batch = np.stack(inference_images)

    # Perform inference on the batch using the model
    model_result = tf.math.softmax(model.predict(batch, verbose=0), axis=-1).numpy()

    # If no object is detected with confidence above the threshold, return empty detections
    if np.all(model_result[:, 1] < confidence_threshold):
        return EMPTY_DETECTIONS

    # Create a dictionary containing the coordinates, classes, and confidence scores of the detected objects
    detection_result = {
        "boxes": tf.ragged.constant(
            [
                [
                    list(box)
                    for box in inference_boxes
                ]
            ],
            ragged_rank=1,
            dtype=tf.float32,
        ),
        "classes": tf.ragged.constant(
            [
                [
                    float(model_result[idx, 1] > confidence_threshold)
                    for idx, _ in enumerate(inference_boxes)
                ]
            ],
            ragged_rank=1,
            dtype=tf.float32,
        ),
        "confidence": tf.ragged.constant(
            [[value for value in model_result[:, 1]]],
            ragged_rank=1,
            dtype=tf.float32,
        ),
    }

    return detection_result

import tensorflow as tf

def format_labels(ground_truth_annotations):
    """
    Returns the ground truth annotations in the format expected by the model.

    Args:
    ground_truth_annotations: list of tuples, the ground truth annotations

    Returns:
    labels: dictionary, the ground truth annotations in the format expected by the model
    """
    # If there are no ground truth annotations, return empty labels
    if len(ground_truth_annotations) == 0:
        return EMPTY_LABELS

    # Create a dictionary with two keys: "boxes" and "classes"
    labels = {
        # "boxes" key contains a ragged tensor with the ground truth bounding boxes
        "boxes": tf.ragged.constant([[list(gt_box) for gt_box in ground_truth_annotations]],
                                     ragged_rank=1,
                                     dtype=tf.float32),
        # "classes" key contains a ragged tensor with the class labels (always 1 in this case)
        "classes": tf.ragged.constant([[1 for _ in ground_truth_annotations]],
                                       ragged_rank=1,
                                       dtype=tf.float32)
    }

    return labels