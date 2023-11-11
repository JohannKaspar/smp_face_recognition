from __future__ import print_function
import cv2
import fire
import time
import tensorflow as tf

from inference_utils import get_sliding_window_detections
from plotting import plot_image_with_bounding_boxes


def detectAndDisplay(image, model):

    input_height = model.input.shape[1]
    input_width = model.input.shape[2]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    detection_boxes = get_sliding_window_detections(image_rgb, (input_height, input_width), model)

    for detection_box in detection_boxes:
        image = cv2.rectangle(image, (detection_box[1], detection_box[0]), (detection_box[3], detection_box[2]), (255, 0, 0), 4) 

    cv2.imshow("Capture - Face detection", image)




def run_simple_model(model_path, camera_device = 0):

    model = tf.keras.models.load_model(model_path)


    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened:
        print("--(!)Error opening video capture")
        exit(0)
    while True:
        _, frame = cap.read()
        if frame is None:
            print("--(!) No captured frame -- Break!")
            break
        detectAndDisplay(frame, model)
        if cv2.waitKey(10) == 27:
            break


def test_simple_model(model_path):

    model = tf.keras.models.load_model(model_path)

    image = cv2.imread("Dataset/originalPics/2002/09/23/big/img_407.jpg")

    while True:
        detectAndDisplay(image, model) 
        time.sleep(0.1)
        if cv2.waitKey(10) == 27:
            break



if __name__ == "__main__":
    fire.Fire({"run": run_simple_model, "test": test_simple_model})