### Introduction
In this lab we will walkthrough on detecting road surface damanage using SSD model and deploy it on Amazon Deeplens. 

Object detection is the process of identifying and localizing objects in an image. A typical object detection solution takes in an image as input and provides a bounding box on the image where an object of interest is, along with identifying what object the box encapsulates. We will be using this model developed in Tensorflow.  We will show you deploy this model
on Deeplens and do inferencing that will detect and classifiy road surface damange.

### The Definition of Road Damage

image []
![Image description](link-to-image)


## First download pre-built model
 
[trainedModel](https://s3-ap-northeast-1.amazonaws.com/mycityreport/trainedModels.tar.gz)

the download file has model files frozen graph of the model (ssd_inception_RoadDamageDetector.pb and ssd_mobilenet_RoadDamageDetector.pb ) and a label map for the model (crack_label_map.pbtxt). We will be using ssd_inception_RoadDamageDetector model.

### Implementation steps

The following sections walk you through the implementation steps in detail.


#### Step 1: Prerequisites
Make sure to register your AWS DeepLens device before you begin. You can follow this ![link](https://youtu.be/j0DkaM4L6n4) for a step-by-step guide to register the device.

TensorFlow 1.5 is not yet installed on the device in Python 2.7. Open terminal of Deeplens or SSH to Deeplens and execute the following command to install TensorFlow.

```python

sudo pip2 install tensorflow==1.5.0

```

If above command doenst work then try following

```python

sudo pip install tensorflow==1.5.0

```

#### Step 2: Upload the tar file to Amazon S3

In this section, you will upload the tar file to Amazon S3 so the AWS DeepLens service can deploy it to the DeepLens device for local inference.

Open the AmazonS3 console.

Create an Amazon S3 bucket in the Northern Virginia Region that ***must contain the term “deeplens”***. The AWS DeepLens default role has permission only to access the bucket with the name containing ”deeplens”. You can name it deeplens-tfmodel-yourinitials

After the bucket is created, upload the tar file to the bucket.

### Create lambda function

In this section, you’ll create a custom Lambda function that will be deployed as part of the AWS DeepLens deployment. This Lambda function contains code to load the custom TensorFlow model that was downloaded from the S3 bucket mentioned previously. This allows you to perform local inferencing (object detection) without connecting back to the AWS Cloud.

Open the AWS Lambda console.
Create a new function from a blueprint, make sure to search for AWS Greengrass and select the Python version.
[Lambda Console](!tensorflow-object-detection-Lambda-console.jpg)

Give the function a meaning name and select AWSDeepLensLambdaRole from Existing role drop-down list. Then create the function. Note: If you do not see this role, make sure that you registered your AWS DeepLens device. The role is automatically created during the registration process.

In the function editor, copy and paste the following code into greengrassHelloWorld.py. Note: Do not rename the file or the function handler, leave everything at the default.

```python
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import greengrasssdk
from threading import Timer
import time
import awscam
from threading import Thread

# Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')

# The information exchanged between IoT and cloud has 
# a topic and a message body.
# This is the topic that this code uses to send messages to cloud
iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
modelPath = "/opt/awscam/artifacts"

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(modelPath,'ssd_inception_RoadDamageDetector.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(modelPath, 'crack_label_map.pbtxt')

def greengrass_infinite_infer_run():
    try:
        # Load the TensorFlow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)
            
        client.publish(topic=iotTopic, payload="Model loaded")
        
        tensor_dict = {}
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes']:
            tensor_name = key + ':0'
            tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)
        #load label map
        label_dict = {}
        with open(PATH_TO_LABELS, 'r') as f:
                id=""
                for l in (s.strip() for s in f):
                        if "id:" in l:
                                id = l.strip('id:').replace('\"', '').strip()
                                label_dict[id]=''
                        if "name:" in l:
                                label_dict[id] = l.strip('name:').replace('\"', '').strip()

        client.publish(topic=iotTopic, payload="Start inferencing")
        while True:
            ret, frame = awscam.getLastFrame()
            if ret == False:
                raise Exception("Failed to get frame from the stream")
            expanded_frame = np.expand_dims(frame, 0)
            # Perform the actual detection by running the model with the image as input
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: expanded_frame})
            scores = output_dict['detection_scores'][0]
            classes = output_dict['detection_classes'][0]
            #only want inferences that have a prediction score of 50% and higher
            msg = '{'
            for idx, val in enumerate(scores):
                if val > 0.5:
                    msg += '"{}": {:.2f},'.format(label_dict[str(int(classes[idx]))], val*100)
            msg = msg.rstrip(',')
            msg +='}'
            
            client.publish(topic=iotTopic, payload = msg)
            
    except Exception as e:
        msg = "Test failed: " + str(e)
        client.publish(topic=iotTopic, payload=msg)

    # Asynchronously schedule this function to be run again in 15 seconds
    Timer(15, greengrass_infinite_infer_run).start()


# Execute the function above
greengrass_infinite_infer_run()


# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    return

```

Save the Lambda function.
To ensure code integrity and consistency, AWS DeepLens only works with the published version of a Lambda function. To publish a Lambda function, choose ***Actions*** and select ***Publish*** new version.

!(image)
Give a description to the published version and choose Publish.



#### Setting up an AWS DeepLens project and deploying a custom TensorFlow object detection model

Open the AWS DeepLens console.

Make sure your DeepLens device is registered before you begin. You can follow this link for a step-by-step guide to register the device.

Make sure TensorFlow 1.5 is installed on the device for Python 2.7. Installation instructions are in prerequisites section.

Make sure you are in the same region where you created the S3 bucket. At the time of writing, AWS DeepLens is only available in the Northern Virginia Region.

In the AWS DeepLens console, in the left navigation panel, choose Models. Then choose Import model.

On the Import model page, choose Externally trained model. In the Model artefact path, enter the S3 bucket and model path you created earlier. Give it a meaningful name and description, and make sure to choose TensorFlow as the model framework. Choose Import model.

image

In the left navigation panel, choose Project. Then choose Create new project.

image

Create a new blank project.

image

Give your project a meaningful name and description, and then choose Add model.

In the AWS DeepLens console, in the left navigation panel, choose Models. Then choose Import model.
On the Import model page, choose Externally trained model. In the Model artefact path, enter the S3 bucket and model path you created earlier. Give it a meaningful name and description, and make sure to choose TensorFlow as the model framework. Choose Import model.

image new

Select the model that you imported earlier, and then choose Add model.

image new

On the project detail page, choose Add function.

image new

Choose the Lambda function you created earlier, then choose Add function.

image new

Choose Create to create the project. On the project page, select the new project that you just set up, and choose Deploy to device.

image new

Select the AWS DeepLens device that you want to deploy the project to, choose Review, and then chose Deploy. The deployment process will take few minutes because the AWS DeepLens device needs to download the model tar file from S3 and the Lambda function to execute locally on the device.
The status pane at the top of the screen displays deployment progress. Wait until it turns green and displays status that the deployment succeeded.

image new

In the Device details pane, copy the MQTT topic, and then navigate to the AWS IoT console.

image new

In the AWS IoT console, in the left navigation pane choose Test, paste the MQTT topic copied in the previous step, then choose Subscribe to topic.

image new

Detected objects and their prediction confidence score are sent in real time through MQTT to the AWS IoT platform.

image new

Conclusion

In this lab, you learned how to deploy a Road Surface Damage detection model to AWS DeepLens. This enables AWS DeepLens to perform real-time object detection using the built-in camera. You can see that it’s possible to use AWS services to build some very powerful AI solutions. 


### Citation:
If you use or find out our dataset useful, please cite our paper in the journal of Computer-Aided Civil and Infrastructure Engineering:

Maeda, H., Sekimoto, Y., Seto, T., Kashiyama, T., & Omata, H. Road Damage Detection and Classification Using Deep Neural Networks with Smartphone Images. Computer‐Aided Civil and Infrastructure Engineering.

@article{maedaroad, title={Road Damage Detection and Classification Using Deep Neural Networks with Smartphone Images}, author={Maeda, Hiroya and Sekimoto, Yoshihide and Seto, Toshikazu and Kashiyama, Takehiro and Omata, Hiroshi}, journal={Computer-Aided Civil and Infrastructure Engineering}, publisher={Wiley Online Library} }

arXiv version is here.
