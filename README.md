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

# Split data for testing on the data that we get the system (_All_data)


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
    
## self,input_tensor objects created
    def call(self, input_tensor):
        x = self.depthwise(input_tensor)
        n1 = self.normalization1(x)
        a1 = self.activation1(n1)
        p = self.pointwise(a1)
        n2 = self.normalization2(p)
        return self.activation2(n2)

class FeatureAttention(tf.keras.layers.Layer):
    
    def __init__(self,filters,pool_size,kernel_size,activation):
        super(FeatureAttention,self).__init__()
        self.concatenate = Concatenate()
        self.multiply = Multiply()
        self.max_pooling = MaxPooling2D(pool_size=pool_size,strides=(1,1),padding='same')
        self.avg_pooling = AveragePooling2D(pool_size=pool_size,strides=(1,1),padding='same')
        self.conv2d = Conv2D(filters=filters,kernel_size=kernel_size,activation=activation, strides=1,padding='same')
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "concatenate": self.concatenate,
            "multiply": self.multiply,
            "max_pooling": self.max_pooling,
            "avg_pooling": self.avg_pooling,
            "2d_convolution": self.conv2d,
        })
        return config
    
    def call(self, input_tensor):
        y=input_tensor
        x = self.concatenate([
            self.max_pooling(input_tensor),
            self.avg_pooling(input_tensor)])
        x = self.conv2d(x)
        return self.multiply([x,y])

# Class Feature Attention 
class FeatureAttention(tf.keras.layers.Layer):
    
    def __init__(self,filters,pool_size,kernel_size,activation):
        super(FeatureAttention,self).__init__()
        self.concatenate = Concatenate()
        self.multiply = Multiply()
        self.max_pooling = MaxPooling2D(pool_size=pool_size,strides=(1,1),padding='same')
        self.avg_pooling = AveragePooling2D(pool_size=pool_size,strides=(1,1),padding='same')
        self.conv2d = Conv2D(filters=filters,kernel_size=kernel_size,activation=activation, strides=1,padding='same')
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "concatenate": self.concatenate,
            "multiply": self.multiply,
            "max_pooling": self.max_pooling,
            "avg_pooling": self.avg_pooling,
            "2d_convolution": self.conv2d,
        })
        return config
    
    def call(self, input_tensor):
        y=input_tensor
        x = self.concatenate([
            self.max_pooling(input_tensor),
            self.avg_pooling(input_tensor)])
        x = self.conv2d(x)
        return self.multiply([x,y])
# Class Residual block


    
    def __init__(self,filters,kernel_size):
        super(ResidualBlock,self).__init__()
        self.conv2d1= Conv2D(filters=filters,kernel_size=kernel_size,strides=(1,1),padding='same')
        self.activation1 = LeakyReLU()
        self.normalization1 = BatchNormalization()
        self.conv2d2 = Conv2D(filters=filters,kernel_size=kernel_size,strides=(1,1),padding='same')
        self.activation2 = LeakyReLU()
        self.normalization2 = BatchNormalization()
        self.concatenate = Concatenate()
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "2d_convolution1": self.conv2d1,
            "2d_convolution2": self.conv2d2,
            "leaky_relu_activation1": self.activation1,
            "leaky_relu_activation2": self.activation2,
            "batch_normalization1": self.normalization1,
            "batch_normalization2": self.normalization2,
            "concatenate": self.concatenate,
            })
        return config
   
    def call(self, input_tensor):
        y = input_tensor
        x = self.conv2d1(input_tensor)
        x = self.normalization1(x)
        x = self.activation1(x)
        x = self.conv2d2(x)
        x = self.normalization2(x)
        x = self.concatenate([x,y])
        return self.activation2(x)



    depthwise_separable1 = DS_Conv(filters=32,kernel_size=(3,3),strides=(2,2))
    depthwise_separable2 = DS_Conv(filters=64,kernel_size=(3,3),strides=(2,2))
    depthwise_separable3 = DS_Conv(filters=128,kernel_size=(3,3),strides=(2,2))
    depthwise_separable4 = DS_Conv(filters=256,kernel_size=(3,3),strides=(2,2))

    feature_Attention1=FeatureAttention(filters=32,pool_size=(3,3),kernel_size=(3,3),activation="sigmoid")
    feature_Attention2=FeatureAttention(filters=64,pool_size=(3,3),kernel_size=(3,3),activation="sigmoid")

    residual_block1=ResidualBlock(filters=32,kernel_size=(3,3))
    residual_block2=ResidualBlock(filters=64,kernel_size=(3,3))

    model = Sequential()

    input_shape = (256,52,1)

    model.add(Conv2D(12,  kernel_size=(3,3), input_shape=input_shape, padding='same', strides=1))
    
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(depthwise_separable1)
    model.add(feature_Attention1)

    model.add(residual_block1)
    model.add(depthwise_separable2)
    model.add(feature_Attention2)

    model.add(residual_block2)
    model.add(depthwise_separable3)
    model.add(depthwise_separable4)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(rate=0.4))
    model.add(Dense(64))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.summary()
    plot_model(model, "my_first_model.png")

# Recall Section of model to testing the accuracy 
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

    model.compile(optimizer=opt,
            loss="categorical_crossentropy",
            metrics=['acc',f1_m])

# Set the checkpoint 
    filepath="checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"## path of your model
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]


    #variables of CNN
    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        shuffle=True,
        callbacks=callbacks_list


# Prediction of data with test size data 

    prediction = model.predict(x_train[:300])
    prediction = model.evaluate(x_test,y_test[:])


# Plotting the accuracy , test , loss ...


    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

