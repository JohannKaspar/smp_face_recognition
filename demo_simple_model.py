from __future__ import print_function
import cv2
import fire
import time
import tensorflow as tf

from inference_utils import get_sliding_window_detections
from plotting import plot_image_with_bounding_boxes

def detectAndDisplay(image, model):
    """
    Detects objects in an image using a sliding window approach and displays the image with bounding boxes around the detected objects.

    Args:
    - image: a numpy array representing the image to be processed
    - model: a pre-trained machine learning model used for object detection

    Returns:
    - None
    """
    size_factors = [1, 2]

    # Get input height and width of the model
    input_height = model.input.shape[1]
    input_width = model.input.shape[2]

    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    # Get detection boxes using sliding window approach
    detection_boxes = get_sliding_window_detections(image_rgb, (input_height, input_width), model, size_factors)

    # Draw bounding boxes around detected objects
    for detection_box in detection_boxes:
        image = cv2.rectangle(image, (detection_box[1], detection_box[0]), (detection_box[3], detection_box[2]), (255, 0, 0), 4) 

    # Display the image with bounding boxes
    cv2.imshow("Capture - Face detection", image)

def run_simple_model(model_path, camera_device = 0):
    """
    Runs a simple model on a video stream from a camera device.

    Args:
    - model_path (str): The path to the saved Keras model.
    - camera_device (int): The index of the camera device to use. Default is 0.

    Returns:
    - None
    """
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)

    # Open the video capture device
    cap = cv2.VideoCapture(camera_device)


    if not cap.isOpened:
        print("--(!)Error opening video capture")
        exit(0)

    # Loop through the frames of the video stream
    while True:
        _, frame = cap.read()

        resized_frame = cv2.resize(frame, (480, 300))

        if resized_frame is None:
            print("--(!) No captured frame -- Break!")
            break
        detectAndDisplay(resized_frame, model)
        if cv2.waitKey(10) == 27:
            break

def test_simple_model(model_path):
    """
    Loads a trained model and continuously applies it to a sample image.

    Args:
        model_path (str): The file path to the trained model.

    Returns:
        None
    """
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load a sample image
    image = cv2.imread("Dataset/originalPics/2002/09/23/big/img_407.jpg")

    # Continuously apply the model to the image
    while True:
        # Apply the model to the image
        detectAndDisplay(image, model) 
        time.sleep(0.1)

        # Break if the user presses the escape key
        if cv2.waitKey(10) == 27:
            break

if __name__ == "__main__":
    # Run the model or test it
    # fire.Fire({"run": run_simple_model, "test": test_simple_model})
    run_simple_model("simple_model.keras")