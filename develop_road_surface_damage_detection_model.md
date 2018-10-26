### Develop road surface damage detection_model

In this lab I will walk you through how to create road surface anomaly detection model using AWS Sagamaker object detection, then make model compatible with AWS Deeplens.

### Objectives are :

1. Collect and annotate dataset
2. Upload dataset to S3
3. Setup AWS Sagemaker
4. Build model
5. Convert model for Deeplens compatibility
6. Save model so that it can be used in Deeplens project.

####
1. Collect and annotate dataset ### *** This is optional step. I already have created dataset and its ready for you to use in Sagemaker. You can download it from here. If you want to create your own dataset then you can follow this step. Otherwise you can download the model and skip to step 2. Upload dataset to S3.

Detecting road surface damage needs detecting multiple objects in the image frame. So we will need to build object detection model.

In order to create this dataset we will need lots of images of roads. Then we need to go through each image and create bounding box where the damage is. We will need to store actual image in image formats such as jpg or gif adn we will also need to bounding box data, i.e. metadata along with the images.

Fortunately this there is already a dataset available for Road damage detection. You can download this dataset from the [link](https://s3-ap-northeast-1.amazonaws.com/mycityreport/RoadDamageDataset.tar.gz).
###### Citation:
{Maeda, H., Sekimoto, Y., Seto, T., Kashiyama, T., & Omata, H. Road Damage Detection and Classification Using Deep Neural Networks with Smartphone Images. Computer‐Aided Civil and Infrastructure Engineering.
}

What if you want to create your own dataset.

In that case you can collect your own raw images various differet ways, manually, crowsourcing, web scraping etc.

Once you have our images, you will need to annotate (or label) the data. 
Once you have your images gathered, the next step is the annotation process. 

There is very good repo you can use to annotate images. [github repository](https://github.com/tzutalin/labelImg). Once you clone the repo, run labelIMg.py python code. This will pop up UI, you may want to start this in sudo. 

``` 
python3 labelIMg.py
```
Through this UI you can draw the bounding box around each object of interest in the image, we saved the annoatations in the same folder as the images. This will be saved in xml file. You will need to do that for all files.

So you can either create your own dataset with annotation or use one cited  earlier [link](https://s3-ap-northeast-1.amazonaws.com/mycityreport/RoadDamageDataset.tar.gz).. 

We will be using [AWS Sagemaker object detection algorithm] (https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/object_detection_pascalvoc_coco/object_detection_image_json_format.ipynb) to build the model
This algorithm needs images to be in designated [json format](https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection.html). 

So we need to convert our xml annotations to json.

I came across very neat [way](https://github.com/bhinthorne) to create json.
First convert all xmls to csv and then convert it to json.
You can use following code. Copy this code and save it xml_to_csv.py. The version we used is below. To change it to fit your data set, simply change "Image Directory" to the directory containing your images and xml annotations, and "Output CSV Name" to the name you want your output csv file to be.

``` python
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
#Change class_dict below to your own. One below are for road surface damage detectionn
	
    class_dict = {'D00':1, 'D01':2, 'D10':3, 'D11':4, 'D20':5, 'D40':6, 'D43':7, 'D44':8, 'D30':9}
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            if member[1].tag =="bndbox":
                value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     class_dict[member[0].text],
                     int(member[1][0].text),
                     int(member[1][1].text),
                     int(member[1][2].text),
                     int(member[1][3].text)
                     )
            else:
                value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     class_dict[member[0].text],
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    
    image_path = os.path.join(os.getcwd(), '.')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('output_name.csv', index=None)
    print('Successfully converted xml to csv.')


main()

```

Now convert labels i csv files to json. 
Use script below I from [from](https://github.com/bhinthorne) 
This script was written specifically for our purposes to take the specific csv files we generated and create the specific format of json files needed for training on Amazon SageMaker.

``` python
import csv
import json

#Just loads all file names into a list for comparison later
def loadFileNames(file_name):
    names = []
    with open(file_name) as rfile:
        reader = csv.DictReader(rfile)
        for row in reader:
            names.append(row['filename'])
    names.append('ENDFILE')
    #print(names)
    return names
#Returns true if next name is the same
def checkNextName(index, names):
    if names[index] == names[index+1]:
        return True
    else:
        return False
#Returns true if the previous name is the same
def checkPrevName(index, names):
    ##If it is the first file return false
    if index == 0:
        return False
    elif names[index] == names[index-1]:
        return True
    else:
        return False

def create_files(file_name, folder):
    class_dict = {1:'D00', 2: 'D01', 3:'D10', 4:'D11', 5:'D20', 6:'D40', 7:'D43', 8:'D44'}
    names = loadFileNames(file_name)
    with open(file_name) as rfile:
        reader = csv.DictReader(rfile)
        counter = 0
        for row in reader:
            #doWrite = True
            #isFirstFile = True
            ##Reading Data
            filename = row['filename']
            width = int(row['width'])
            height = int(row['height'])
            depth = 3
            class_id = int(row['class'])
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            box_width = abs(xmax-xmin)
            box_height = abs(ymax-ymin)
            ##Done Reading Data
            print(filename)
            print(class_id)
            '''
                If the previous name is not the same, clear data and load the new data
            '''
            if not checkPrevName(counter, names):
                ## Clearing Data
                data = {}
                ## Loading Data
                data['file'] = filename
                data['image_size'] = []
                data['image_size'].append({
                'width':width,
                'height':height,
                'depth':depth
                })
                data['annotations'] = []
                data['annotations'].append({
                    'class_id':class_id,
                    'left':xmin,
                    'top':ymin,
                    'width':box_width,
                    'height':box_height
                })

                data['categories'] = []
                data['categories'].append({
                    'class_id':class_id,
                    'name': class_dict[class_id]
                })
            '''
                If the previous file name is the same, append annotation data
            '''
            if checkPrevName(counter, names):
                ##Appending Annotations
                data['annotations'].append({
                    'class_id':class_id,
                    'left':xmin,
                    'top':ymin,
                    'width':box_width,
                    'height':box_height
                })
            '''
                If the next name is not the same then we want to write the data
            '''    
            if not checkNextName(counter, names):
                if filename[-4:] == "jpeg":
                    new_name = filename[:-5]
                else:
                    new_name = filename[:-4]
                output_loc = 'json_files_test'+new_name+'.json'
                with open(output_loc, 'w') as outfile:
                    json.dump(data, outfile)
            counter += 1

create_files("output_name.csv", "train")
```

You should now have a folder containing all the json files with annotations for each image! The final step to preprocess the data is to split your images and annoatations into train and validation folders. First, create folders train, validation, train_annotation, and validation_annoatation. Then split up your images and json files into train and validation categories. You want many more training images than validation images. I manually copied split training and validation images to 70-30%. Put your training images into the train directory, and the corresponding json files into the train_annoation directory. Do the same with your validation data for the validation and validation_annoataion directories. Your data is now all preprocessed! The next step is to move onto AWS to begin training. 

Now your dataset is ready for model building.

#### Step 2. Uploading to AWS


Create an AWS S3 bucket to store the dataset, Then copy folders train, validation, train_annotation, and validation_annoatation to the S3. Here is [guide](https://docs.aws.amazon.com/quickstarts/latest/s3backup/step-1-create-bucket.html) to help you create S3 bucket.

To create an S3 Bucket, navigate to the S3 storage services and select "Create Bucket". Enter the bucket name, our name for the first model that worked was "deeplens-roaddamagedetection". Be sure to select the region that is the same as the IAM role you are using, and where your notebook instance will be running.
We first need to create a the folder "roaddamagedetection". 
If you downloaded the dataset tar file from my link above then untar or unzip the file. You will see four folders rain, validation, train_annoation, and validation_annotation. Uploade thse folders inside "roaddamagedetection" folder.

If you created the dataset mannually then upload your train, validation, train_annoation, and validation_annotation folders under "roaddamagedetection" folder. We now have all the data for our custom model uploaded into an S3 Bucket.

###  Step 3. Setup AWS Sagemaker

### **Creating a notebook instnace for training**

To create a notebook instance for training, navigate to SageMaker from AWS services and go to the Notebook instances tab. Chose "Create Notebook Instance" and give your notebook isntance a name. We left all other settings as they were, but for the IAM role we used an existing "AmazonSageMaker-ExecutionRole". 

It may take few minutes for notebook to spin-up. 

### Step 4. Build the model

You can use the jupyter notebook I already created. You can clone or download this git. 
With Sagemaker you can access terminal. You can use terminal to install your own libraries or clone git hub.
To open terminal. On Sagemaker Jupyter notebook, click "New" on the left side, it will popup box
![Sagemaker terminal](images/terminal.jpg)
Once terminal is open. Clone git as below
```
git clone https://github.com/mahendrabairagi/visual.git

```

This notebook
## Modifying the sample notebook


### **Set Up**

The only change we made is setting 'bucket = YOUR BUCKET NAME' to point to the bucket where our training data is stored. For us this looked like:

``` python
bucket = 'deeplens-xxxxxx-xxxxxx'
prefix = 'roaddamagedetection'
```


We then ran the three cells under the "Set Up" section. The next section titled "Data Preperation" is not needed for our training. This section is downloading the PascalVOC dataset that the example is built to train on. Since we have already created our own data, we do not need to train on this data. If we were to train on PascalVOC data rather than our own custom data, we could just run the notebook straight without any preprocessing or uploading of the data.


### **Training**

Under section titled training, this is where we create our model with hyperparameters and begin the training. The first cell creates the sagemaker.estimator.Estimator object which will be used for training. We did not make any changes to this cell, thus its contents were:

``` python
od_model = sagemaker.estimator.Estimator(training_image,
                                         role, 
                                         train_instance_count=1, 
                                         train_instance_type='ml.p3.16xlarge',
                                         train_volume_size = 50,
                                         train_max_run = 360000,
                                         input_mode = 'File',
                                         output_path=s3_output_location,
                                         sagemaker_session=sess)
```

The next cell contains the hyperparameters for training. A description of the hyperparameters used by SageMaker's built in training algorith can be found [here](https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection-api-config.html?shortFooter=true).

``` python
od_model.set_hyperparameters(base_network='resnet-50',
                             use_pretrained_model=1,
                             num_classes=8,
                             mini_batch_size=8,
                             epochs=2,
                             learning_rate=0.004,
                             lr_scheduler_step='10',
                             lr_scheduler_factor=0.1,
                             optimizer='sgd',
                             momentum=0.9,
                             weight_decay=0.0005,
                             overlap_threshold=0.5,
                             nms_threshold=0.45,
                             image_shape=600,
                             label_width=600,
                             num_training_samples=6661)
```

now run .fit to train the model. This may take while depending on epochs annd other hyper parameters you choose.
``` python
od_model.fit(inputs=data_channels, logs=True)
```
At first a string of dots will show up and begin progressing, eventually a lot of red text will show up documenting the training job. Once the training job is complete, you should see a message that looks like this:  

```
===== Job Complete =====
Billable seconds: 556
```

### Step 5. Convert model for Deeplens compatibility

Now open up a terminal in the DeepLens. You can either ssh into the DeepLens or connect to a monitor and work with a keyboard and mouse. Once you have a terminal open, change directories into /opt/awscam/artifacts. If your project was sucessfully deployed you should see the same three files in there that were outputted by your SageMaker training job. The next step is to optimize the model and create the xml file our lambda function points to for inference. Feel free to remove the project from the device at this point, the files will remain loaded on the DeepLens and we will be redeploying the project later.

### **Optimizing the Model for DeepLens**

To create an xml file for inference we have to use an AWS module called "mo" and run an mo.optimize function. This function takes our json files outputted from the training job, now deployed onto the DeepLens, and creates an xml file used for inference. After spending a long time struggling and trying to get the optimizer working on our model, we were informed that an SSD model trained on SageMaker cannot be directly deployed or optimized on the DeepLens. Apparenlty there are some artifacts within the files containing the model that are not supported by the DeepLens. To fix this issue, we had to run the artifacts from the model training through a "deploy.py" script from this [github repo](https://github.com/apache/incubator-mxnet/tree/master/example/ssd). 

To run deploy.py we first cloned the github repo from the DeepLens terminal with the command: 

```bash
sudo git clone https://github.com/apache/incubator-mxnet
```

Once we cloned the directory we then moved "model_algo_1-0000.params and model_algo_1-symbol.json into the same directory as deploy.py.  Run deploy.py with the following command:

```bash
sudo python3 deploy.py  --network='resnet50' --epoch=0 --num-class=8  --prefix='model_algo_1' --data-shape=600

```

Be sure to adjust the paramaters to fit your trained model. If you run that command successfully you should see a message that looks like:

Saved Model: 'model_name'.params
Saved Model: 'model_name'.json

Your model artifacts are now ready to be optimized. We then moved the .params and .json files back to their original directory in /opt/awscam/artifacts. Again be sure to move these files, do not copy them. Now we can use the mo.optimize function mentioned earlier to generate the xml file. Return to the home directory in the DeepLens terminal, and open python with the sequence of commands:

```bash
cd ~
sudo python
```

If you wish to see the details about the model optimizer refer to the [aws documentation](https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-model-optimizer-api.html). We have provided the commands that worked for us. Once you have python opene, run these commands to optimize your model and generate the xml file:

```python
import mo
model_name = 'your_model_name'
input_height = your_input_height
input_width = your_input_width
mo.optimize(model_name, input_height, input_width)
```

By default this will generate an xml  and bin file with the name 'your_model_name'.xml and 'your_model_name'.bin in the /opt/awscam/artifacts directory. 


### 6. Save model so that it can be used in Deeplens project.

You can compress (i.e. tar) these two files (.xml and .bin) and upload it to a S3 bucket.

```
sudo tar -zcvf model.tar.gz 'your_model_name'.xml and 'your_model_name'.bin

```

When you create Deeplens project you will need a model. To setup model in Deeplens, you can give path of this S3 bucket where the model.tar.gz file is stored.

### Conclusion

In this walkthrough, you learned how to build dataset, build object detection model and how to convert model so that it can be deployed on Deeplens.
Next step, you can try [lab](https://github.com/mahendrabairagi/visual/blob/master/roaddamageinspection.md) that shows how to deploy this model on Deeplens.

