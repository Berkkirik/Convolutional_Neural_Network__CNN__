# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:16:05 2022

@author: Berk
"""
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from keras.utils.vis_utils import plot_model
from tensorflow.keras import models, Model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, ReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, GlobalMaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation, DepthwiseConv2D, Add, Multiply, LeakyReLU, Flatten
from tensorflow.keras import activations
from tensorflow.keras.callbacks import ModelCheckpoint
from DatasetReader import fixed_data_read_twostation
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder
from scipy import signal
from keras import backend as K
import datetime
import os
import pickle

import matplotlib.pyplot as plt





warnings.filterwarnings("ignore")

labels = ["walking","fall"] #labels (ex => labels = ["loc1","loc2"])
folder_name = "path/to/file" #directory of the data (ex => folder_name = "data/fall-walking")
sample_number = 1600 #sample number (ex => sample_number = 480)

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





# amp1,amp2,phase1,phase2,y=fixed_data_read_twostation(folder_name,sample_number,labels,True)

amp1,amp2,y=get_data(folder_name,labels)


# # # Once read the data save it as np.array so no need to read each time 
np.save('amp1',amp1)
np.save('amp2',amp2)
# np.save('phase1',phase1)
# np.save('phase2',phase2)
np.save('y',y)

y = np.load('y.npy')
amp1 = np.load('amp1.npy')
amp2 = np.load('amp2.npy')
# phase1 = np.load('phase1.npy')
# phase2 = np.load('phase1.npy')




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
    # print("mod,",mod_range,"length, ",timelength)
    for i in range(0,timelength-step_size,step_size):
        # print("i ",i)
        start=i
        end=i+winSize
        # print("Start:",i," End:",i+winSize)
        add_temp_data=data[i:end,:]
        amp_segmented.append(add_temp_data)
        y_segmented.append(y_class)
        
    class_counter=class_counter+1    
    
    
amp_segmented=np.asarray(amp_segmented)
y_segmented=np.asarray(y_segmented)
train_data=amp_segmented.copy()
train_labels=y_segmented.copy()
train_labels=train_labels.reshape((-1,1))


# STEP 2 
# PREPROCESS DATA USING BUTTERWORTH
# PREPROCESS DATA USING GAUSSIAN SMOOTHING


#convert y to categorical
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(train_labels)

train_labels = enc.transform(train_labels)

#BUTTERWORTH FILTER

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



#you can chose order=5,cutoff_freq = 20,freq=100
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









#STEP 3 DATA SPLIT

x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size = 0.2)

# from sklearn.preprocessing import StandardScaler

# x_train=x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
# x_test=x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))

# transformer = StandardScaler().fit(x_train)
# x_train=transformer.transform(x_train)
# x_test=transformer.transform(x_test)

# x_train=x_train.reshape((x_train.shape[0],int(x_train.shape[1]/52),52))
# x_test=x_test.reshape((x_test.shape[0],int(x_test.shape[1]/52),52))




#STEP 4 CREATE MODEL

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

class ResidualBlock(tf.keras.layers.Layer):
    
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
# model.add(Dropout(rate=0.2))
model.add(BatchNormalization())
model.add(ReLU())
model.add(depthwise_separable1)
model.add(feature_Attention1)
# model.add(Dropout(rate=0.4))
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

# checkpoint
filepath="checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"## path of your model
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    callbacks=callbacks_list
)

prediction = model.predict(x_train[:300])
prediction = model.evaluate(x_test,y_test[:])

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









# inter_output_model = tf.keras.Model(model.input, model.get_layer(index = -2).output )
# inter_output = inter_output_model.predict(x_test)


# ##To more specifically,
# # inter_output_model = keras.Model(model.input, model.get_layer(index = 3).output ) assuming the intermedia layer is indexed at 3.
# # To use it,
# # inter_output = inter_output_model.predict(x_test) x_test is the data you feed into your model


 
# from sklearn.manifold import TSNE
# #for original test data features before train
# t_org=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
# X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(t_org)


# #for model test data features after train
# inter_output = inter_output_model.predict(x_test)
# t_train=inter_output.copy()
# train_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(t_train)




# ##for original train data features before train
# t_org=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
# X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(t_org)


# #for model train data features after train
# inter_output = inter_output_model.predict(x_train)
# t_train=inter_output.copy()
# train_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(t_train)

# # import matplotlib

# #for test data
# fig, (ax1,ax2)=plt.subplots(1,2)
# y_test=yt = np.argmax (y_test, axis = 1)
# ax1.scatter(X_embedded[:,0],X_embedded[:,1],c=y_test)
# ax2.scatter(train_embedded[:,0],train_embedded[:,1],c=y_test)

# # import matplotlib
# #for train data
# fig, (ax1,ax2)=plt.subplots(1,2)
# y_train=yt = np.argmax (y_train, axis = 1)
# ax1.scatter(X_embedded[:,0],X_embedded[:,1],c=y_train)
# ax1.legend(["nobody","loc1","loc2"])
# ax2.scatter(train_embedded[:,0],train_embedded[:,1],c=y_train)


#---------------------------save the whole model------------------------
# date_time = str(datetime.datetime.now())
# date_time = date_time.replace(" ",",")
# date_time = date_time.replace(":",".")

# model_name = "model_" + date_time
# path = "models" + "/" + model_name

# if not os.path.exists(path):
#     os.makedirs(path)
# model.save(path)

# converter = tf.lite.TFLiteConverter.from_keras_model(model) 
# tflite_model = converter.convert()
# if not os.path.exists(path+'/tflite/'):
#     os.makedirs(path+'/tflite/')
# with open(path + '/tflite/' + model_name + '.tflite', 'wb') as f:
#   f.write(tflite_model)

# #save the standart scaler
# filename = "scaler_" + date_time + ".pkl"
# pickle.dump(transformer, open(path + '/' + filename, 'wb'))

# #STEP 5 TEST CASE
# #READ TEST DATA FROM ANOTHER FOLDER
# #AND TEST THE MODEL

# labels = ["nobody","loc1","loc2"] #labels (ex => labels = ["loc1","loc2"])
# folder_name = "data/test_data_loc" #directory of the data (ex => folder_name = "data/loc1-loc2")
# sample_number = 480 #sample number (ex => sample_number = 480)

# amp1,amp2,phase1,phase2,y=fixed_data_read_twostation(folder_name,sample_number,labels,True)

# winSize = 256
# step_size = 128

# i=0
# amp_segmented=[]
# y_segmented=[]
# class_counter=0
# for data in amp1:
#     y_class=y[class_counter]
#     mod_range=len(data)%winSize
#     timelength=len(data)-mod_range
#     # print("mod,",mod_range,"length, ",timelength)
#     for i in range(0,timelength-step_size,step_size):
#         # print("i ",i)
#         start=i
#         end=i+winSize
#         # print("Start:",i," End:",i+winSize)
#         add_temp_data=data[i:end,:]
#         amp_segmented.append(add_temp_data)
#         y_segmented.append(y_class)
        
#     class_counter=class_counter+1    


# amp_segmented=np.asarray(amp_segmented)
# y_segmented=np.asarray(y_segmented)
# train_data=amp_segmented.copy()
# train_labels=y_segmented.copy()
# train_labels=train_labels.reshape((-1,1))


# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

# enc = enc.fit(train_labels)

# train_labels = enc.transform(train_labels)

# #BUTTERWORTH FILTER
# # from scipy import signal
# # import matplotlib.pyplot as plt


# # order = 5
# # cutoff_freq = 20
# # freq=100
# # time = np.linspace(0,1, 256, endpoint=False)
# # normalized_cutoff_freq = 2 * cutoff_freq / freq
# # numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq)

# # butter_filtered_data=[]
# # for data in train_data.copy():
# #     for i in range(data.shape[1]):
# #         # print(i)
# #         sample_signal=data[:,i]
# #         filtered_signal = signal.lfilter(numerator_coeffs, denominator_coeffs, sample_signal)
# #         #filtered_signal=filtered_signal.reshape((len(filtered_signal),1))
# #         data[:,i]=filtered_signal
# #     butter_filtered_data.append(data)
# # butter_filtered_data=np.asarray(butter_filtered_data)



# x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size = 0.1)

# from sklearn.preprocessing import StandardScaler

# x_train=x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
# x_test=x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))

# transformer = StandardScaler().fit(x_train)


# x_train=transformer.transform(x_train)
# x_test=transformer.transform(x_test)

# x_train=x_train.reshape((x_train.shape[0],int(x_train.shape[1]/52),52))
# x_test=x_test.reshape((x_test.shape[0],int(x_test.shape[1]/52),52))



# prediction = model.evaluate(x_test,y_test[:])
# prediction = model.predict(x_train)