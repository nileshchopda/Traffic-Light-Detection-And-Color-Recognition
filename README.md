# Traffic Light Detection And Color Recognition
Traffic Light Detection using Tensorflow Object Detection API and Microsoft COCO Dataset

## Introduction
Humans can easily detect and identify objects present in an image frame. The human visual system is fast and accurate and can perform complex tasks like identifying multiple objects and detect obstacles with little conscious thought and within less time. With the availability of large amounts of data, faster GPUs, and better algorithms, we can now easily train machines to detect and classify multiple objects within an image with high accuracy.

With the advancements in technology, there has been a rapid increase in the development of autonomous vehicles and smart cars. Accurate detection and recognition of traffic lights is a crucial part in the development of such cars. The concept involves enabling autonomous vehicles to automatically detect traffic lights using the least amount of human interaction. Automating the process of traffic light detection in cars would also help to reduce accidents as machines do better jobs than humans.

## Implementation Strategy
The experiment was implemented using transfer learning of the Microsoft's Common Objects in Context (COCO) pre-trained models and Tensorflow's Object Detection API.The COCO dataset contains images of 90 classes ranging from bird to baseball bat. The first 14 classes are all related to transportation, including bicycle, car, and bus, etc. The ID for traffic light is 10.For the classes included in COCO dataset, please see 'mscoco_label_map.pbtxt'.

TensorFlow’s Object Detection API is a powerful tool that makes it easy to construct, train, and deploy object detection models. In most of the cases, training an entire convolutional network from scratch is time consuming and requires large datasets. This problem can be solved by using the advantage of transfer learning with a pre-trained model using the TensorFlow API.They have released different versions detection models trained on MS COCO dataset,from which,I have selected 2 models to test my experiment.The selection of these models is based on mAP,mean Average Precision,which indicates how well the model performed on the COCO dataset.Generally models that take longer to compute perform better.

Once the object is detected in the image frame ,it then crops image and extracts only object's frame which is further processed to recognize the dominant color in the object frame.For this experiment,we have detected Red and yellow colors in objects's frame and not green color.The reason for detecting Red or Yellow is that car has to take Stop action whenever there is Red or Yellow light on Traffic Light.The default action for this experiment is 'Go' action.             
