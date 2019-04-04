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

#### Function to Load Image into Numpy Array
For processing image frames it needs to be in form which machines can understand easily.So we will convert image frame into numpy array and further use it process and detect traffic lights.Function to convert image into numpy array is:
```
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
```

#### Function to Read Traffic Light Objects

Here,we will write a function to detect Traffic Light objects and crop object box of the image to recognize color inside the object. We will create a stop flag,which it will use to take the actions based on recognized color of the traffic light.This flag will be True for Red and Yellow colors and False otherwise.Function to read traffic light objects is:
```
def read_traffic_lights_object(image, boxes, scores, classes, max_boxes_to_draw=20, min_score_thresh=0.5,
                               traffic_ligth_label=10):
    im_width, im_height = image.size
    stop_flag = False
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores[i] > min_score_thresh and classes[i] == traffic_ligth_label:
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            crop_img = image.crop((left, top, right, bottom))

            if detect_red_and_yellow(crop_img):
                stop_flag = True

    return stop_flag
```

#### Function to Plot Detected Image
To visualize the results ,we will plot original image along with detected object boxes,scores and actions to take .Function to plot detected image is:
```
def plot_origin_image(image_np, boxes, classes, scores, category_index):
    # Size of the output images.
    IMAGE_SIZE = (12, 8)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        min_score_thresh=.5,
        use_normalized_coordinates=True,
        line_thickness=3)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)

    # save augmented images into hard drive
    # plt.savefig( 'output_images/ouput_' + str(idx) +'.png')
    plt.show()
```

#### Function to Detect Traffic Lights and to Recognize Color
To detect traffic lights and draw bounding boxes around the traffic lights,we need to specify directory paths for test images,model to be used,frozen detection graph file,lapbel map etc.This function will detect traffic lights and draw bounding boxes around them.It will also print Stop or Go based on values of stop_flag.If detected,it will change commands flag to true.
```
def detect_traffic_lights(PATH_TO_TEST_IMAGES_DIR, MODEL_NAME, Num_images, plot_flag=False):
    """
    Detect traffic lights and draw bounding boxes around the traffic lights
    :param PATH_TO_TEST_IMAGES_DIR: testing image directory
    :param MODEL_NAME: name of the model used in the task
    :return: commands: True: go, False: stop
    """

    # --------test images------
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'img_{}.jpg'.format(i)) for i in range(1, Num_images + 1)]

    commands = []

    # What model to download
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'mscoco_label_map.pbtxt'

    # number of classes for COCO dataset
    NUM_CLASSES = 90

    # --------Download model----------
    if path.isdir(MODEL_NAME) is False:
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())

    # --------Load a (frozen) Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # ---------Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)

                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                stop_flag = read_traffic_lights_object(image, np.squeeze(boxes), np.squeeze(scores),
                                                       np.squeeze(classes).astype(np.int32))
                if stop_flag:
                    # print('{}: stop'.format(image_path))  # red or yellow
                    commands.append(False)
                    cv2.putText(image_np, 'Stop', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                else:
                    # print('{}: go'.format(image_path))
                    commands.append(True)
                    cv2.putText(image_np, 'Go', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

                # Visualization of the results of a detection.
                if plot_flag:
                    plot_origin_image(image_np, boxes, classes, scores, category_index)

    return commands
```

#### Detect Traffic Lights in test_images directory

To detect traffic lights specify Number of images,test images directory path and Model name.Here we used "faster_rcnn_resnet101_coco_11_06_2017" model. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.You can download and choose models based on your requirements.Lets Detect Traffic Lights in test_images directory : 
```
if __name__ == "__main__":
    # Specify number of images to detect
    Num_images = 5

    # Specify test directory path
    PATH_TO_TEST_IMAGES_DIR = './test_images'

    # Specify downloaded model name
    # MODEL_NAME ='ssd_mobilenet_v1_coco_11_06_2017'    # for faster detection but low accuracy
    MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'  # for improved accuracy

    commands = detect_traffic_lights(PATH_TO_TEST_IMAGES_DIR, MODEL_NAME, Num_images, plot_flag=True)
    print(commands)  # commands to print action type, for 'Go' this will return True and for 'Stop' this will return False

```
    
## Results

<img src="output_images\Figure_2.png" >
    
<img src="output_images\Figure_5.png" >

<img src="output_images\Figure_7.png" >

## Simulation Result

<img src="final_5ca5a1cb87e4f90013e225e5_555664.gif">

## Training Custom Objects

As Traffic Light is one of the classes trained using Tensorflow Object Detection API and Micosoft's COCO dataset ,we have skipped the process of training custom models.But If you want to train and detect custom objects, you can refer following tutorials on training your custom object detection classifier :

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md


## Limitations

This project was an attempt to perform object detection and color recognition on COCO dataset. The resulting model can be run directly on small devices like **Raspberry Pi** using TensorFlow Lite format.
To perform real time object detection on live video stream,the machine must be equipped with hardware capable to perform high computations quickly.This can be achived using high end Graphical Processing Unit which have multiple cores to perform parallel processing for object detection.   


## Future Study

The broad objective of Object detection system is to make autonomous vehicles interpret surrounding environment and take actions better than humans.Self-driving cars are rapidly evolving as we see unimaginable innovation in hardware, software, and computing capabilities.But, cars have to perform object detection in real-time in order to detect objects approaching quickly and avoid them. There must be a very low latency time for high accuracy, meaning that very high computing and graphical power is needed. We’ll need to improve the power of our processor units in order to implement computer vision safely for autonomous vehicles.

However, we also need a very accurate model upwards of 99.9%, since any mistakes made can be disastrous and cost human lives. Our current models have not achieved such high accuracies yet and we must generate more data to train on or design even better models. There are other better models for object detection like Faster-RCNNs but are still far from the accuracy we need. If we can improve object detection greatly, we’ll be one step closer to self-driving cars, and a safer and more convenient future.


## Expansion of Project

Object detection classifiers can work on multiple objects and thus I have expanded scope of project to traffic surveillance and video analysis:  
Models trained on COCO dataset have facility of detecting ~90 classes.This can be used to detect multiple objects from image frame and perform analysis on objects detected.Analysis includes number of detected objects and their bounding boxes.This type of projects can be useful in efficient traffic management.Here is an example showing object detection of multiple objects and their analysis.


<img src="final_5ca5a26f8f0f530014d040c8_74490.gif">

## References

https://github.com/nileshchopda/Traffic-Light-Detection-And-Color-Recognition
https://www.pyimagesearch.com/2018/05/14/a-gentle-guide-to-deep-learning-object-detection/
https://software.intel.com/en-us/articles/traffic-light-detection-using-the-tensorflow-object-detection-api
https://medium.com/comet-app/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852
https://towardsdatascience.com/evolution-of-object-detection-and-localization-algorithms-e241021d8bad



Hope you enjoyed the read!


