import keras_cv
import numpy as np
import tensorflow as tf


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
    n_tilde = img_dim // dim
    overlap = int(((n_tilde + 1) * dim - img_dim) / n_tilde)

    box_begins = []
    for i in range(n_tilde + 1):
        bgn = i * (dim - overlap)

        if i == n_tilde:
            bgn = img_dim - dim
        box_begins.append(bgn)

    return box_begins


def get_sliding_window_inference_boxes(image_shape, input_shape):
    
    image_height, image_width, _ = image_shape
    input_height, input_width = input_shape

    inference_boxes = []

    for h_min in get_sliding_window_corners(image_height, input_height):
        for w_min in get_sliding_window_corners(image_width, input_width):
            inference_boxes.append(
                (h_min, w_min, h_min + input_height, w_min + input_width)
            )

    return inference_boxes


def get_sliding_window_detections(img_array, input_shape, model):
    inference_boxes = get_sliding_window_inference_boxes(
        img_array.shape, input_shape
    )  # (input_height, input_width)

    inference_images = []
    for inference_box in inference_boxes:
        inference_images.append(
            img_array[
                inference_box[0] : inference_box[2], inference_box[1] : inference_box[3]
            ]
        )

    batch = np.stack(inference_images)

    model_result = tf.math.argmax(
        tf.math.softmax(model.predict(batch), axis=-1), axis=-1
    ).numpy()

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
    "NOTE: This function assumes that there is only one class besides the background!"

    inference_boxes = get_sliding_window_inference_boxes(
        img_array.shape, input_shape
    )  # (input_height, input_width)

    inference_images = []
    for inference_box in inference_boxes:
        inference_images.append(
            img_array[
                inference_box[0] : inference_box[2], inference_box[1] : inference_box[3]
            ]
        )

    batch = np.stack(inference_images)

    model_result = tf.math.softmax(model.predict(batch, verbose=0), axis=-1).numpy()

    if np.all(model_result[:, 1] < confidence_threshold):
        return EMPTY_DETECTIONS

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

def format_labels(ground_truth_annotations):
    if len(ground_truth_annotations) == 0:
        return EMPTY_LABELS

    labels = {
        "boxes": tf.ragged.constant([[list(gt_box) for gt_box in ground_truth_annotations]],
        ragged_rank=1,
        dtype=tf.float32,
    ),
        "classes": tf.ragged.constant([[1 for _ in ground_truth_annotations]],
        ragged_rank=1,
        dtype=tf.float32,
    )
    }

    return labels