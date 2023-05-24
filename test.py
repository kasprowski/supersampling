test_dir = 'test_data_VD1'

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
import errors as er

loaded_models = {}
models = ('upsampling_slow_hm','upsampling_fast_hm','transpose_slow_hm','transpose_fast_hm','upsampling_slow_ht','upsampling_fast_ht','transpose_slow_ht','transpose_fast_ht')
for m in models:
    model = tf.keras.models.load_model(f'{m}.h5')
    loaded_models[m] = model


for test_dir in ('test_data_hm','test_data_004','test_data_ht','test_data_VD1','test_data_VD1_2','test_data_G1','test_data_G2'):
    print("+"*40)
    print(test_dir)
    fixations2 = []
    saccades2 = []
    avg_fix = 0
    avg_sac = 0
    file_no =0
    files = {}
    for file in os.listdir(test_dir):
    #    print(test_dir+'/'+file)
        errors4f = {}
        data = uf.load_file(test_dir+'/'+file)
        f,s = uf.prepare_fix_sac(data,40,64)
        fs,fl = uf.create_datasets(f)
        fsamples = fs[64]
        flabels = fl[64]
        ss,sl = uf.create_datasets(s)
        ssamples = ss[64]
        slabels = sl[64]
        
        for m in models:
            model = loaded_models[m]
            if 'slow' in m:
                error = er.calc_errors(model,fsamples,flabels)[0]
            else:    
                error = er.calc_errors(model,ssamples,slabels)[0]
            errors4f[m] = error        
            #print(file,m,error,sep='\t')
            print("*",end='')
        files[file] = errors4f
    
    #print(files)
    print('')    
    avg_errors = {}
    
    for m in models:
        avg_errors[m] = 0
        
    for file in files:
    #    print
        fname = file.replace('_','.')[:-4]
        errors = files[file]
        print(f"{fname}& {errors['upsampling_slow_hm']:.4f} & {errors['upsampling_fast_hm']:.4f} & {errors['transpose_slow_hm']:.4f} & {errors['transpose_fast_hm']:.4f} & {errors['upsampling_slow_ht']:.4f} & {errors['upsampling_fast_ht']:.4f} & {errors['transpose_slow_ht']:.4f} & {errors['transpose_fast_ht']:.4f} \\\\");
        for e in errors:
            err = errors[e]
            avg_errors[e] = avg_errors[e] + err
        
    for e in avg_errors:
        avg_errors[e] = avg_errors[e] / 8        
    
    
    errors = avg_errors
    print(f"average& {errors['upsampling_slow_hm']:.4f} & {errors['upsampling_fast_hm']:.4f} & {errors['transpose_slow_hm']:.4f} & {errors['transpose_fast_hm']:.4f} & {errors['upsampling_slow_ht']:.4f} & {errors['upsampling_fast_ht']:.4f} & {errors['transpose_slow_ht']:.4f} & {errors['transpose_fast_ht']:.4f} \\\\");
    print("+"*40)

#    {error_fix[0]:.5f}& {error_sac[0]:.5f}\\\\")
    
"""
    fname = file.replace('_','.')[:-4]
    print(f"{fname}& {error_fix[0]:.5f}& {error_sac[0]:.5f}\\\\")
    avg_fix += error_fix[0]
    avg_sac += error_sac[0]
    file_no += 1
    files[file]=errors
"""
#print(f"average& {avg_fix/file_no:.5f}& {avg_sac/file_no:.5f}")


    

