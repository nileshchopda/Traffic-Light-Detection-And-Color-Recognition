# Traffic Light Detection And Color Recognition
Traffic Light Detection using Tensorflow Object Detection API and Microsoft COCO Dataset

## Introduction
Humans can easily detect and identify objects present in an image frame. The human visual system is fast and accurate and can perform complex tasks like identifying multiple objects and detect obstacles with little conscious thought and within less time. With the availability of large amounts of data, faster GPUs, and better algorithms, we can now easily train machines to detect and classify multiple objects within an image with high accuracy.

With the advancements in technology, there has been a rapid increase in the development of autonomous vehicles and smart cars. Accurate detection and recognition of traffic lights is a crucial part in the development of such cars. The concept involves enabling autonomous vehicles to automatically detect traffic lights using the least amount of human interaction. Automating the process of traffic light detection in cars would also help to reduce accidents as machines do better jobs than humans.

## Implementation Strategy
The experiment was implemented using transfer learning of the Microsoft's Common Objects in Context (COCO) pre-trained models and Tensorflow's Object Detection API.The COCO dataset contains images of **90 classes** ranging from bird to baseball bat. The first 14 classes are all related to transportation, including bicycle, car, and bus, etc. The ID for traffic light is 10.For the classes included in COCO dataset, please see **'mscoco_label_map.pbtxt'.**

TensorFlow’s Object Detection API is a powerful tool that makes it easy to construct, train, and deploy object detection models. In most of the cases, training an entire convolutional network from scratch is time consuming and requires large datasets. This problem can be solved by using the advantage of transfer learning with a pre-trained model using the TensorFlow API.They have released different versions detection models trained on MS COCO dataset,from which,I have selected 2 models to test my experiment.The selection of these models is based on mAP,**mean Average Precision**,which indicates how well the model performed on the COCO dataset.Generally models that take longer to compute perform better.

Once the object is detected in the image frame ,it then crops image and extracts only object's frame which is further processed to recognize the dominant color in the object frame.For this experiment,we have detected Red and yellow colors in objects's frame and not green color.The reason for detecting Red or Yellow is that car has to take Stop action whenever there is Red or Yellow light on Traffic Light.The default action for this experiment is 'Go' action.             

## Installation
Before starting with the experiment,lets understand dependancies which we need to take care of before installing Tensorflow Object Detection API.Tensorflow Object Detection API depends on the following libraries:
  -python3
  -Protobuf 3.0.0
  -Python-tk
  -Pillow 1.0
  -lxml
  -tf Slim (which is included in the "tensorflow/models/research/" checkout)
  -Jupyter notebook
  -Matplotlib
  -Tensorflow (>=1.12.0)
  -Cython
  -contextlib2
  -cocoapi

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
Download the [cocoapi](https://github.com/cocodataset/cocoapi) and copy the pycocotools subfolder to the tensorflow/models/research directory if you are interested in using COCO evaluation metrics. The default metrics are based on those used in Pascal VOC evaluation. To use the COCO object detection metrics add metrics_set: "coco_detection_metrics" to the eval_config message in the config file. To use the COCO instance segmentation metrics add metrics_set: "coco_mask_metrics" to the eval_config message in the config file.
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow>/models/research/
```

### Protobuf Installation/Compilation
The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be downloaded and compiled.

This should be done as follows:

Head to the [protoc releases page](https://github.com/protocolbuffers/protobuf/releases)

Download the latest *-win32.zip release (e.g. protoc-3.5.1-win32.zip)

Create a folder in C:\Program Files and name it Google Protobuf.

Extract the contents of the downloaded *-win32.zip, inside C:\Program Files\Google Protobuf

Add C:\Program Files\Google Protobuf\bin to your Path environment variable

In a new Anaconda/Command Prompt, cd into TensorFlow/models/research/ directory and run the following command:
```
# From within TensorFlow/models/research/
Get-ChildItem object_detection/protos/*.proto | foreach {protoc "object_detection/protos/$($_.Name)" --python_out=.}
```
