# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:27:24 2021

@author: ASUS
"""
#dataset reader

import pandas as pd
import os
import numpy as np
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import Counter

class Dataset_reader():
    def __init__(self,folder_name,sample_number):
        self.folder_name = folder_name
        self.sample_number = sample_number

def fixed_data_read_twostation(foldername,sample_number,labels,should_drop):
    #example usage is    
    # folder_name="data/loc1-loc2"
    # sample_number=480
    # labels=["loc1","loc2"]
    # df_sta1_amp,df_sta2_amp,df_sta1_phase,df_sta2_phase=fixed_data_read_twostation(folder_name,sample_number,labels)
    subcarrier_num = 64
    df_sta1_amp = []
    df_sta2_amp = []
    df_sta1_phase = []
    df_sta2_phase = []
    dfy = []

    drops = ["0","1","2","3","4","5","32","59","60","61","62","63"]
    names = []
    for i in range(subcarrier_num):
        names.append(str(i))

    folder_name = ""+foldername
    file_list = os.listdir(folder_name)
    sample_number = sample_number
    labels = labels
    print("DR: 44")

    for f in file_list:
        tokens = f.split("_")
        if tokens[0] in labels:
            print("DR: 49")
            if tokens[3]=="STA1" and tokens[4]=="amplitude" :
                print("DR: 51")
                filename_string_amplitude_sta1 = ""+tokens[0]+"_"+tokens[1]+"_"+tokens[2]+"_"+"STA1"+"_"+"amplitude"+"_"+tokens[5]+"_"+tokens[6]
                filename_string_amplitude_sta2 = ""+tokens[0]+"_"+tokens[1]+"_"+tokens[2]+"_"+"STA2"+"_"+"amplitude"+"_"+tokens[5]+"_"+tokens[6]
                filename_string_phase_sta1 = ""+tokens[0]+"_"+tokens[1]+"_"+tokens[2]+"_"+"STA1"+"_"+"phase"+"_"+tokens[5]+"_"+tokens[6]
                filename_string_phase_sta2 = ""+tokens[0]+"_"+tokens[1]+"_"+tokens[2]+"_"+"STA2"+"_"+"phase"+"_"+tokens[5]+"_"+tokens[6]
                
                print("new data")
                
                #sta1 amp
                df_temp1 = None
                df_temp1 = pd.read_csv(folder_name+"/"+filename_string_amplitude_sta1,sep=":",names=names,index_col=False)
                if should_drop:
                    df_temp1 = df_temp1.drop(columns="0")
                    df_temp1.columns = names[0:63]
                    df_temp1["63"] = 0
                    df_temp1=df_temp1.drop(columns=drops)
                if df_temp1.shape[0]<sample_number:
                    print("skipped")
                    continue
                
                df_temp1 = df_temp1.replace(np.nan, 0)
                temp1=df_temp1.values[:sample_number]
                
                
                #sta2 amp
                df_temp2=None
                df_temp2=pd.read_csv(folder_name+"/"+filename_string_amplitude_sta2,sep=":",names=names,index_col=False)
                if should_drop:
                    df_temp2 = df_temp2.drop(columns="0")
                    df_temp2.columns = names[0:63]
                    df_temp2["63"] = 0
                    df_temp2 = df_temp2.drop(columns=drops)

                if df_temp2.shape[0]<sample_number:
                    print("skipped")
                    continue
                
                df_temp2 = df_temp2.replace(np.nan, 0)
                temp2 = df_temp2.values[:sample_number]
                
    
                #sta1 phase
                df_temp3 = None
                df_temp3 = pd.read_csv(folder_name+"/"+filename_string_phase_sta1,sep=":",names=names,index_col=False)
                if should_drop:
                    df_temp3 = df_temp3.drop(columns="0")
                    df_temp3.columns = names[0:63]
                    df_temp3["63"] = 0
                    df_temp3 = df_temp3.drop(columns=drops)
                if df_temp3.shape[0]<sample_number:
                    print("skipped")
                    continue
                
                df_temp3 = df_temp3.replace(np.nan, 0)
                temp3 = df_temp3.values[:sample_number]
                
    
                #sta2 phase
                df_temp4 = None
                df_temp4 = pd.read_csv(folder_name+"/"+filename_string_phase_sta2,sep=":",names=names,index_col=False)
                if should_drop:
                    df_temp4 = df_temp4.drop(columns="0")
                    df_temp4.columns = names[0:63]
                    df_temp4["63"] = 0
                    df_temp4 = df_temp4.drop(columns=drops)
                if df_temp4.shape[0]<sample_number:
                    print("skipped")
                    continue
                
                df_temp4 = df_temp4.replace(np.nan, 0)
                temp4 = df_temp4.values[:sample_number]
                
                
                df_sta1_amp.append(temp1)
                df_sta2_amp.append(temp2)
                df_sta1_phase.append(temp3)
                df_sta2_phase.append(temp4)
                print("append")
                
                #add y values to list
                print(tokens[0])
                dfy.append(labels.index(tokens[0]))
                print("DR: 133")
                        
    df_sta1_amp = np.asarray(df_sta1_amp,dtype=float)
    df_sta2_amp = np.asarray(df_sta2_amp,dtype=float)
    df_sta1_phase = np.asarray(df_sta1_phase,dtype=float)
    df_sta2_phase = np.asarray(df_sta2_phase,dtype=float)
    dfy = np.asarray(dfy,dtype=float)
    print("DR: 140")
    return df_sta1_amp,df_sta2_amp,df_sta1_phase,df_sta2_phase,dfy



def fixed_data_read_continuos_twostation(foldername,sample_number,labels,should_drop):
    #exapmle usage is    
    # folder_name="train_data/six_activities_5sec"
    # sample_number=50
    # labels=["fall","walking","nobody","sitting"]
    # df_sta1_amp,df_sta2_amp,df_sta1_phase,df_sta2_phase=fixed_data_read_twostation(folder_name,sample_number,labels)
    subcarrier_num=64
    df_sta1_amp= []
    df_sta2_amp= []
    df_sta1_phase= []
    df_sta2_phase= []
    dfy=[]

    drops=["0","1","2","3","4","5","59","60","61","62","63"]
    names=[]
    for i in range(subcarrier_num):
        names.append(str(i))

    # df_sta1_amp= pd.DataFrame(columns=names)
    # df_sta1_amp=df_sta1_amp.drop(columns=drops)

    folder_name=""+foldername
    file_list = os.listdir(folder_name)
    sample_number=sample_number
    labels=labels




    #labels.index("walking")
    for f in file_list:
        tokens=f.split("_")
        if tokens[0] in labels:
                   
            if tokens[3]=="sta1" and tokens[2]=="amplitude" :
                
                filename_string_amplitude_sta1=""+tokens[0]+"_"+tokens[1]+"_amplitude_sta1_"+tokens[4]
                filename_string_amplitude_sta2=""+tokens[0]+"_"+tokens[1]+"_amplitude_sta2_"+tokens[4]
                filename_string_phase_sta1=""+tokens[0]+"_"+tokens[1]+"_phase_sta1_"+tokens[4]
                filename_string_phase_sta2=""+tokens[0]+"_"+tokens[1]+"_phase_sta2_"+tokens[4]
                
                #sta1 amp
                df_temp=None
                df_temp=pd.read_csv(folder_name+"/"+filename_string_amplitude_sta1,sep=":",names=names,index_col=False)
                if should_drop:
                    #print("shape--",df_temp.shape)
                    #print(df_temp.loc[0])
                    df_temp=df_temp.drop(columns="0")
                    #print("shape",df_temp.shape)
                    df_temp.columns=names[0:63]
                    df_temp["63"]=0
                    df_temp=df_temp.drop(columns=drops)
                    print("--shape",df_temp.shape)
                print(df_temp.shape)
                if df_temp.shape[0]<sample_number:
                    print("skipped")
                    continue
                
                print("not contiune")
                #df_temp.drop("index",inplace=True, axis=1)
                df_temp=df_temp.replace(np.nan, 0)
                d=df_temp.shape[0]%50
                df_temp=df_temp.values[:-d]
                df_temp=np.reshape( df_temp,(int(df_temp.shape[0]/sample_number),sample_number , df_temp.shape[1]))
                print("shape",df_temp.shape)
                for data in df_temp:
                    df_sta1_amp.append(data)
                    dfy.append(labels.index(tokens[0]))
                    
                # temp=df_temp.values[:sample_number]
                # df_sta1_amp.append(temp)
                
                
                
                #sta2 amp
                df_temp=None
                df_temp=pd.read_csv(folder_name+"/"+filename_string_amplitude_sta2,sep=":",names=names,index_col=False)
                if should_drop:
                    df_temp=df_temp.drop(columns="0")
                    df_temp.columns=names[0:63]
                    df_temp["63"]=0
                    df_temp=df_temp.drop(columns=drops)
                print(df_temp.shape)
                if df_temp.shape[0]<sample_number:
                    print("skipped")
                    continue
                
                df_temp=df_temp.replace(np.nan, 0)
                d=df_temp.shape[0]%50
                df_temp=df_temp.values[:-d]
                df_temp=np.reshape( df_temp,(int(df_temp.shape[0]/sample_number),sample_number , df_temp.shape[1]))

                for data in df_temp:
                    df_sta2_amp.append(data)
                    # dfy.append(labels.index(tokens[0]))
                # print(tokens[0])
                # dfy.append(labels.index(tokens[0]))
                    
    
                
                # #sta1 phase
                # df_temp=None
                # df_temp=pd.read_csv(folder_name+"/"+filename_string_phase_sta1,sep=":",names=names,index_col=False)
                # if should_drop:
                #     df_temp=df_temp.drop(columns="0")
                #     df_temp.columns=names[0:63]
                #     df_temp["63"]=0
                #     df_temp=df_temp.drop(columns=drops)
                # print(df_temp.shape)
                # if df_temp.shape[0]<sample_number:
                #     print("skipped")
                #     continue
                
                # print("not contiune")
                # df_temp=df_temp.replace(np.nan, 0)
                # temp=df_temp.values[:sample_number]
                # df_sta1_phase.append(temp)
    
                
                # #sta2 phase
                # df_temp=None
                # df_temp=pd.read_csv(folder_name+"/"+filename_string_phase_sta2,sep=":",names=names,index_col=False)
                # if should_drop:
                #     df_temp=df_temp.drop(columns="0")
                #     df_temp.columns=names[0:63]
                #     df_temp["63"]=0
                #     df_temp=df_temp.drop(columns=drops)
                # print(df_temp.shape)
                # if df_temp.shape[0]<sample_number:
                #     print("skipped")
                #     continue
                
                # print("not contiune")
                # df_temp=df_temp.replace(np.nan, 0)
                # temp=df_temp.values[:sample_number]
                # df_sta2_phase.append(temp)
                
                # #add y values to list
                # print(tokens[0])
                # dfy.append(labels.index(tokens[0]))
            
    df_sta1_amp=np.asarray(df_sta1_amp,dtype=float)
    df_sta2_amp= np.asarray(df_sta2_amp,dtype=float)
    df_sta1_phase=[]
    df_sta2_phase=[]
    # df_sta1_phase= np.asarray(df_sta1_phase,dtype=float)
    # df_sta2_phase= np.asarray(df_sta2_phase,dtype=float)
    dfy=np.asarray(dfy,dtype=float)
    return df_sta1_amp,df_sta2_amp,df_sta1_phase,df_sta2_phase,dfy

def fixed_data_read_continuos_twostation_slided(foldername,sample_number,labels,should_drop,slide):
    #exapmle usage is    
    # folder_name="train_data/six_activities_5sec"
    # sample_number=50
    # labels=["fall","walking","nobody","sitting"]
    # df_sta1_amp,df_sta2_amp,df_sta1_phase,df_sta2_phase=fixed_data_read_twostation(folder_name,sample_number,labels)
    subcarrier_num=64
    df_sta1_amp= []
    df_sta2_amp= []
    df_sta1_phase= []
    df_sta2_phase= []
    dfy=[]

    drops=["0","1","2","3","4","5","59","60","61","62","63"]
    names=[]
    for i in range(subcarrier_num):
        names.append(str(i))

    # df_sta1_amp= pd.DataFrame(columns=names)
    # df_sta1_amp=df_sta1_amp.drop(columns=drops)

    folder_name=""+foldername
    file_list = os.listdir(folder_name)
    sample_number=sample_number
    labels=labels




    #labels.index("walking")
    for f in file_list:
        tokens=f.split("_")
        print(tokens[0] )
        if tokens[0] in labels:
                   
            if tokens[3]=="sta1" and tokens[2]=="amplitude" :
                
                filename_string_amplitude_sta1=""+tokens[0]+"_"+tokens[1]+"_amplitude_sta1_"+tokens[4]
                filename_string_amplitude_sta2=""+tokens[0]+"_"+tokens[1]+"_amplitude_sta2_"+tokens[4]
                filename_string_phase_sta1=""+tokens[0]+"_"+tokens[1]+"_phase_sta1_"+tokens[4]
                filename_string_phase_sta2=""+tokens[0]+"_"+tokens[1]+"_phase_sta2_"+tokens[4]
                
                #sta1 amp
                df_temp=None
                df_temp=pd.read_csv(folder_name+"/"+filename_string_amplitude_sta1,sep=":",names=names,index_col=False)
                if should_drop:
                    #print("shape--",df_temp.shape)
                    #print(df_temp.loc[0])
                    df_temp=df_temp.drop(columns="0")
                    #print("shape",df_temp.shape)
                    df_temp.columns=names[0:63]
                    df_temp["63"]=0
                    df_temp=df_temp.drop(columns=drops)
                #     print("--shape",df_temp.shape)
                # print(df_temp.shape)
                if df_temp.shape[0]<sample_number:
                    print("skipped")
                    continue
                
                # print("not contiune")
                #df_temp.drop("index",inplace=True, axis=1)
                df_temp=df_temp.replace(np.nan, 0)
                d=df_temp.shape[0]%50
                df_temp=df_temp.values[:-d]
                
                #apply slide process
                start=0
                df_slided=[]
                for i in range (sample_number,len(df_temp),slide):
                    #print(start,i)
                    start=start+slide
                    df_slided.append(df_temp[start:i])


                for data in df_slided:
                    df_sta1_amp.append(data)
                    dfy.append(labels.index(tokens[0]))
                    
                # temp=df_temp.values[:sample_number]
                # df_sta1_amp.append(temp)
                
                
                
                #sta2 amp
                df_temp=None
                df_temp=pd.read_csv(folder_name+"/"+filename_string_amplitude_sta2,sep=":",names=names,index_col=False)
                if should_drop:
                    df_temp=df_temp.drop(columns="0")
                    df_temp.columns=names[0:63]
                    df_temp["63"]=0
                    df_temp=df_temp.drop(columns=drops)
                # print(df_temp.shape)
                if df_temp.shape[0]<sample_number:
                    print("skipped")
                    continue
                
                df_temp=df_temp.replace(np.nan, 0)
                d=df_temp.shape[0]%50
                df_temp=df_temp.values[:-d]
                
                #apply slide process
                start=0
                df_slided=[]
                for i in range (sample_number,len(df_temp),slide):

                    start=start+slide
                    df_slided.append(df_temp[start:i])


                for data in df_slided:
                    df_sta2_amp.append(data)

                    
    
                
                # #sta1 phase
                # df_temp=None
                # df_temp=pd.read_csv(folder_name+"/"+filename_string_phase_sta1,sep=":",names=names,index_col=False)
                # if should_drop:
                #     df_temp=df_temp.drop(columns="0")
                #     df_temp.columns=names[0:63]
                #     df_temp["63"]=0
                #     df_temp=df_temp.drop(columns=drops)
                # print(df_temp.shape)
                # if df_temp.shape[0]<sample_number:
                #     print("skipped")
                #     continue
                
                # print("not contiune")
                # df_temp=df_temp.replace(np.nan, 0)
                # temp=df_temp.values[:sample_number]
                # df_sta1_phase.append(temp)
    
                
                # #sta2 phase
                # df_temp=None
                # df_temp=pd.read_csv(folder_name+"/"+filename_string_phase_sta2,sep=":",names=names,index_col=False)
                # if should_drop:
                #     df_temp=df_temp.drop(columns="0")
                #     df_temp.columns=names[0:63]
                #     df_temp["63"]=0
                #     df_temp=df_temp.drop(columns=drops)
                # print(df_temp.shape)
                # if df_temp.shape[0]<sample_number:
                #     print("skipped")
                #     continue
                
                # print("not contiune")
                # df_temp=df_temp.replace(np.nan, 0)
                # temp=df_temp.values[:sample_number]
                # df_sta2_phase.append(temp)
                
                # #add y values to list
                # print(tokens[0])
                # dfy.append(labels.index(tokens[0]))
            
    df_sta1_amp=np.asarray(df_sta1_amp,dtype=float)
    df_sta2_amp= np.asarray(df_sta2_amp,dtype=float)
    df_sta1_phase=[]
    df_sta2_phase=[]
    # df_sta1_phase= np.asarray(df_sta1_phase,dtype=float)
    # df_sta2_phase= np.asarray(df_sta2_phase,dtype=float)
    dfy=np.asarray(dfy,dtype=float)
    return df_sta1_amp,df_sta2_amp,df_sta1_phase,df_sta2_phase,dfy
