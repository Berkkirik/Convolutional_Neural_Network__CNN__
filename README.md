# Convolutional_Neural_Network__CNN__
 Deep Learning Algorithm with Dataset_Reader and Datasets 

 CNN is used for  ; CNN (Convolutional Neural Network) is a type of neural network commonly used for processing and analyzing image data. It is particularly effective for tasks such as image recognition, object detection, and classification.

CNNs are designed to automatically learn and identify patterns and features within image data by using convolutional layers that apply filters to the input image. These filters detect edges, corners, and other visual features that are then used to classify and identify objects within the image.

In addition to image processing, CNNs are also used in natural language processing (NLP) tasks such as text classification and sentiment analysis. They are also applied in other domains such as speech recognition, time-series analysis, and recommendation systems.


#Explanation Of Code
## İMPORT SECTİON 
---To Run Code , you need to install the requirement libraries ---> For example : pip install tensorflow (on cmd)
---After setting up the env. (requirement libraries) , you need to import the libraries that we installed 
__[Find, install and publish Python packages with the Python Package Index]__ (https://pypi.org/)


----Example Usage 
import tensorflow as tf 
import numpy as np

# get_Data
*labels* , *folder_name* , *sample_number* these are important parameter for this algorithm 
*labels* = "walking" , "fall"
*sample_number* = "300"
*folder name* = path/to/file

def get_data(path, labels):
#def get_data(file_list, labels):
    file_list = os.listdir(path)
    labels = labels
    d_train1=[]
    d_train2 = []
    d_trainy=[]
    drops = [0,1,2,3,4,5,32,59,60,61,62,63]
    sample_number = 300
    
    for f in file_list:
        tokens = f.split("_")
        train_data=None
        #print(tokens)
        if tokens[3]=="STA1":
            try:
                with open(path +"/"+f, 'rb') as f:
                      train_data=np.load(f)
                      
                drops = [0,1,2,3,4,5,32,59,60,61,62,63]
                train_data=np.delete(train_data,drops,1)
                if train_data.shape[0]==sample_number:
                    d_trainy.append(labels.index(tokens[0]))
                d_train1.append(train_data)
            except ValueError:
                print(f"{f} has 0 size")
                continue
        if tokens[3] =="STA2":
            try:
                with open(path +"/"+f, 'rb') as f:
                      train_data=np.load(f)  
                drops = [0,1,2,3,4,5,32,59,60,61,62,63]
                train_data=np.delete(train_data,drops,1)
                # if train_data.shape[0]==sample_number:
                #     d_trainy.append(labels.index(tokens[0]))
                d_train2.append(train_data)
            except ValueError:
                print(f"{f} has 0 size")
                continue
            
    return np.array(d_train1), np.array(d_train2), np.array(d_trainy)
###__Above cod___ It explains the reading protocol that includes codebooks from the past. For more information you can check DatasetReader.py file.

# LOAD SECTİON
### amp1,amp2,phase1,phase2,y=fixed_data_read_twostation(folder_name,sample_number,labels,True)

amp1,amp2,y=get_data(folder_name,labels)


### Once read the data save it as np.array so no need to read each time 
np.save('amp1',amp1)
np.save('amp2',amp2)
np.save('y',y)
y = np.load('y.npy')
amp1 = np.load('amp1.npy')
amp2 = np.load('amp2.npy')


# PARAMETERS AND SEGMENTATİON
## "win_size" and "step_size" are significant parameters on model , Also amp_segmented is function that take some parameters came from "win_size"and "step_Size".
winSize = 256
step_size = 128

i=0
amp_segmented=[]
y_segmented=[]
class_counter=0
for data in amp1:
   
    y_class=y[class_counter]
    
    mod_range=len(data)%winSize
    
    timelength=len(data)-mod_range
   
   for i in range(0,timelength-step_size,step_size):
        
        start=i
        end=i+winSize
        
        
        add_temp_data=data[i:end,:]
        amp_segmented.append(add_temp_data)
        y_segmented.append(y_class)
    class_counter=class_counter+1    
    
    
amp_segmented=np.asarray(amp_segmented)
y_segmented=np.asarray(y_segmented)
train_data=amp_segmented.copy()
train_labels=y_segmented.copy()
train_labels=train_labels.reshape((-1,1))


# Convert y to Categorial