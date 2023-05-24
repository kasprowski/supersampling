import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, Conv2D,MaxPooling1D, BatchNormalization, Input
from tensorflow.keras.layers import UpSampling1D, LeakyReLU, Conv1DTranspose, Concatenate, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv1DTranspose
import math
from tensorflow.keras.initializers import RandomNormal

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


FACTOR=8
import util_functions as uf
import models

for train_set in ('hm','ht'):
    for model_type in ('upsampling','transpose'):
        for event_type in ('slow','fast'):
            print("+"*40)
            train_dir = 'train_data_'+train_set
            model_name = model_type +"_"+event_type+"_"+train_set+".h5"
            
            print("train_dir",train_dir)
            print("model_name",model_name)
            
            if model_type == 'upsampling':
                model = models.upsampling_model((8,2),FACTOR)
            if model_type == 'transpose':
                model = models.transpose_model((8,2),FACTOR)
            
            model.compile(loss='mse', optimizer="adam",metrics=['mae'])
            
            
            fixations = []
            saccades = []
            for file in os.listdir(train_dir):
                print(train_dir+'/'+file)
                data = uf.load_file(train_dir+'/'+file)
                f,s = uf.prepare_fix_sac(data,40,64)
                fixations.extend(f)
                saccades.extend(s)
            
            print("fix num",len(fixations))
            print("sac num",len(saccades))
            
            if event_type == 'slow':
                ds,dl = uf.create_datasets(fixations,factor=FACTOR,durations=[64])
            if event_type == 'fast':
                ds,dl = uf.create_datasets(saccades,factor=FACTOR,durations=[64])
            
            trainSamples = ds[64]
            trainLabels = dl[64]
            #np.savez_compressed('sac_train_dir',trainSamples,trainLabels)
            print("train samples size: ",trainSamples.shape[0])
            
            for e in range(100):
                H = model.fit(trainSamples, trainLabels, epochs=10, verbose=0)
                print(f"epoch {e}. loss = {H.history['loss'][-1]}")
            
            model.save(model_name)
            print("saved in ",model_name)