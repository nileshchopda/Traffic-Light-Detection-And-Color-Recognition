# Traffic Light Detection And Color Recognition
Traffic Light Detection using Tensorflow Object Detection API and Microsoft COCO Dataset


## Introduction
Humans can easily detect and identify objects present in an image frame. The human visual system is fast and accurate and can perform complex tasks like identifying multiple objects and detect obstacles with little conscious thought and within less time. With the availability of large amounts of data, faster GPUs, and better algorithms, we can now easily train machines to detect and classify multiple objects within an image with high accuracy.

With the advancements in technology, there has been a rapid increase in the development of autonomous vehicles and smart cars. Accurate detection and recognition of traffic lights is a crucial part in the development of such cars. The concept involves enabling autonomous vehicles to automatically detect traffic lights using the least amount of human interaction. Automating the process of traffic light detection in cars would also help to reduce accidents as machines do better jobs than humans.


## Working Strategy
The experiment was implemented using transfer learning of the Microsoft's Common Objects in Context (COCO) pre-trained models and Tensorflow's Object Detection API.The COCO dataset contains images of **90 classes** ranging from bird to baseball bat. The first 14 classes are all related to transportation, including bicycle, car, and bus, etc. The ID for traffic light is 10.For the classes included in COCO dataset, please see **'mscoco_label_map.pbtxt'.**

TensorFlow’s Object Detection API is a powerful tool that makes it easy to construct, train, and deploy object detection models. In most of the cases, training an entire convolutional network from scratch is time consuming and requires large datasets. This problem can be solved by using the advantage of transfer learning with a pre-trained model using the TensorFlow API.They have released different versions detection models trained on MS COCO dataset,from which,I have selected 2 models to test my experiment.The selection of these models is based on mAP,**mean Average Precision**,which indicates how well the model performed on the COCO dataset.Generally models that take longer to compute perform better.

Once the object is detected in the image frame ,it then crops image and extracts only object's frame which is further processed to recognize the dominant color in the object frame.For this experiment,we have detected Red and yellow colors in objects's frame and not green color.The reason for detecting Red or Yellow is that car has to take Stop action whenever there is Red or Yellow light on Traffic Light.The default action for this experiment is 'Go' action.             


## Installation
Before starting with the experiment,lets understand dependancies which we need to take care of before installing Tensorflow Object Detection API.Tensorflow Object Detection API depends on the following libraries:
```
python3
Protobuf 3.0.0
Python-tk
Pillow 1.0
lxml
tf Slim (which is included in the "tensorflow/models/research/" checkout)
Jupyter notebook
Matplotlib
Tensorflow (>=1.12.0)
Cython
contextlib2
cocoapi
```

A typical user can install Tensorflow using one of the following commands depending on hardware configuration:
```
# For CPU
pip install tensorflow
# For GPU (works best on machines with nvidia GPU installed on it)
pip install tensorflow-gpu
```
The remaining libraries can be installed using conda install or pip:
```
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib
```

### COCO API installation
Download the [cocoapi](https://github.com/cocodataset/cocoapi) and install it using pip command mentioned below:
```
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```


### Protobuf Installation/Compilation
The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be downloaded and compiled.

This should be done as follows:

  -Head to the [protoc releases page](https://github.com/protocolbuffers/protobuf/releases)

  -Download the latest *-win32.zip release (e.g. protoc-3.5.1-win32.zip)

  -Create a folder in C:\Program Files and name it Google Protobuf.

  -Extract the contents of the downloaded *-win32.zip, inside C:\Program Files\Google Protobuf

  -Add C:\Program Files\Google Protobuf\bin to your Path environment variable

  -In a new Anaconda/Command Prompt, cd into TensorFlow/models/research/ directory and run the following command:
  ```
  # From within TensorFlow/models/research/
  for /f %i in ('dir /b object_detection\protos\*.proto') do protoc object_detection\protos\%i --python_out=.
  ```

NOTE: You MUST open a new Anaconda/Command Prompt for the changes in the environment variables to take effect.


### Adding necessary Environment Variables
 -As Tensorflow\models\research\object_detection is the core package for object detection, it’s convenient to add the specific folder to our environmental variables.
 ```
 The following folder must be added to your PYTHONPATH environment variable (Environment Setup):

<PATH_TO_TF>\TensorFlow\models\research\object_detection
```
 

**NOTE** : For different operating systems,Installation steps are different.To check detailed installation guidelines for your system ,please  refer 
 https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md


## Implementation of Code
Before starting the actual working of Code,we need to import important libraries:


#### Import Important Libraries

```
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from os import path
from utils import label_map_util
from utils import visualization_utils as vis_util
import time
import cv2
```


#### Function to detect Red and Yellow Color
To Detect color from the detected traffic light object frame will need frame to be in masked form.So we will convert and mask image with Red and Yellow color masks.Here we will let machine to take 'Stop' action once it detect area of detected Red or Yellow color is more than threshold set,otherwise the default action taken is 'Go' for which we need not detect Green color. So,let's write a function to detect Red and Yellow color from image : 
```
def detect_red_and_yellow(img, Threshold=0.01):
    """
    detect red and yellow
    :param img:
    :param Threshold:
    :return:
    """

    desired_dim = (30, 90)  # width, height
    img = cv2.resize(np.array(img), desired_dim, interpolation=cv2.INTER_LINEAR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red1 = np.array([170, 70, 50])
    upper_red1 = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)

    # defining the Range of yellow color
    lower_yellow = np.array([21, 39, 64])
    upper_yellow = np.array([40, 255, 255])
    mask2 = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    # red pixels' mask
    mask = mask0 + mask1 + mask2

    # Compare the percentage of red values
    rate = np.count_nonzero(mask) / (desired_dim[0] * desired_dim[1])

    if rate > Threshold:
        return True
    else:
        return False
        
```

#### Function to load image into Numpy Array
For processing image frames it needs to be in form which machines can understand easily.So we will convert image frame into numpy array and further use it process and detect traffic lights.Function to convert image into numpy array is:
```

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
        
```

