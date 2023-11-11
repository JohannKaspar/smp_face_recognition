# Face detection

## Preparation

* Create conda environment:

    ```bash
    conda create -n smp python=3.9
    ```
    
* Activate environment and install requirements:

    ```bash
    pip install -r requirements.txt
    ```
    
* Download dataset from `http://vis-www.cs.umass.edu/fddb/` and extract to folder `Dataset/originalPics`

* Download labels from `http://vis-www.cs.umass.edu/fddb/` and extract in folder `Dataset`

* Download `README.txt` from `http://vis-www.cs.umass.edu/fddb/` as it contains useful information about parsing the annotations

## Binary clasifier

Please note that the purpose of the following code is to provide a skeleton that allows to get started with face detection more quickly, not to provide a great model! Identifying the shortcomings of the provided solution can give some inspiration on what to improve when developing an own model.

### Preparation

In order to extract training data for a binary classifier (deciding whether a patch contains a face or not), open and run `extract_faces.ipynb`. This will extract face images using the bounding box annotations (as foreground images) and image patches without faces (as background images). 

### Model training

These extracted images can be used to train a simple classifier. Open the notebook `train_simple_model.ipynb`, which contains a very simple training script for a very lightweight convolutional neural network, using Tensorflow as deep learning framework.

TODO: Visualize generated embedding mit UMAP

### Inference
Open the notebooks `inference.ipynb` or `inference_improved.ipynb`. The first contains some visualizations of how the model trained in the previous section performs in a sliding window fashion. The second notebook is somewhat more advanced and contains also the computation of some metrics.

## Demos
The scripts `demo_opencv_viola_jones.py` and `demo_simple_model.py` contain demo code for two algorithms: The Viola-Jones algorithm (which is implemented in OpenCV for face and eye detection) and the simple algorithm that can be trained with the provided code above. These demo scripts are best run on computers with webcam.
