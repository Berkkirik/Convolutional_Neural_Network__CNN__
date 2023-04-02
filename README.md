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

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(train_labels)

train_labels = enc.transform(train_labels)


# BUTTERWORTH FILTER
## You can reach butterworth filter documentation from **https://en.wikipedia.org/wiki/Butterworth_filter**
order = 5
cutoff_freq = 20
freq=100
time = np.linspace(0,1, 256, endpoint=False)
normalized_cutoff_freq = 2 * cutoff_freq / freq
numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq)

butter_filtered_data=[]
for data in train_data.copy():
    for i in range(data.shape[1]):
        # print(i)
        sample_signal=data[:,i]
        filtered_signal = signal.lfilter(numerator_coeffs, denominator_coeffs, sample_signal)
        #filtered_signal=filtered_signal.reshape((len(filtered_signal),1))
        data[:,i]=filtered_signal
    butter_filtered_data.append(data)
butter_filtered_data=np.asarray(butter_filtered_data)

# Butterworth Low pass Filtering section
## You can reach butterworth low pass filter documentation from **https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7 ** ,  **https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units**


def  low_pass_filter(train_data,order,freq,cutoff_freq):
    

    time = np.linspace(0,1, train_data.shape[1], endpoint=False)
    normalized_cutoff_freq = 2 * cutoff_freq / freq
    numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq)
    butter_filtered_data=[]
    for data in train_data.copy():
        for i in range(data.shape[1]):
            # print(i)
            sample_signal=data[:,i]
            filtered_signal = signal.lfilter(numerator_coeffs, denominator_coeffs, sample_signal)
            #filtered_signal=filtered_signal.reshape((len(filtered_signal),1))
            data[:,i]=filtered_signal
        butter_filtered_data.append(data)
    butter_filtered_data=np.asarray(butter_filtered_data)
    return butter_filtered_data

x_train=low_pass_filter(train_data,5,100,20)



# STEP 3 DATA SPLIT

x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size = 0.2)

#
Split data for testing on the data that we get the system (_All_data)


# STEP 4 CREATE MODEL


## For more information (https://www.tensorflow.org/api_docs/python/tf/keras)


class DS_Conv(tf.keras.layers.Layer):
    
    def __init__(self,filters,kernel_size,strides):
        super(DS_Conv,self).__init__()
        self.depthwise = DepthwiseConv2D(kernel_size=kernel_size,strides=strides,padding="same")
        self.pointwise = Conv2D(filters=filters,kernel_size=(1,1),padding="same")
        self.activation1 = ReLU()
        self.activation2 = ReLU()
        self.normalization1 = BatchNormalization()
        self.normalization2 = BatchNormalization()
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "depthwise_convolution": self.depthwise,
            "pointwise_convolution": self.pointwise,
            "relu_activation1": self.activation1,
            "relu_activation2": self.activation2,
            "batch_normalization1": self.normalization1,
            "batch_normalization2": self.normalization2,
        })
        return config
    
## self,input_tensor functions created
    def call(self, input_tensor):
        x = self.depthwise(input_tensor)
        n1 = self.normalization1(x)
        a1 = self.activation1(n1)
        p = self.pointwise(a1)
        n2 = self.normalization2(p)
        return self.activation2(n2)
