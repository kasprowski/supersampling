import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, Conv2D,MaxPooling1D, BatchNormalization, Input
from tensorflow.keras.layers import UpSampling1D, LeakyReLU, Conv1DTranspose, Concatenate, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv1DTranspose

from tensorflow.keras.initializers import RandomNormal

import math
def downsample(data,factor):
    newdata = []
    for i in range(0,data.shape[0],factor):
        newdata.append(data[i])
    return np.array(newdata)


def upsample(data,factor):
    newdata = []
    for i in range(data.shape[0]):
        for j in range(factor):
            newdata.append(data[i])
    return np.array(newdata)

def upsample_reg(data,factor):
    newdata = []
    for i in range(data.shape[0]-1):
        start = data[i]
        end = data[i+1]
        diff = data[i+1]-data[i]
        diff = diff/factor
        #print(start,end,diff)
        for j in range(factor):
            newdata.append(data[i]+diff*j)
    for j in range(factor):
        newdata.append(data[data.shape[0]-1])
    return np.array(newdata)


def loss(data1,data2):
    #if data1.shape[0]!=data1.shape[0]:
    diff = 0
    for i in range(data1.shape[0]):
        dx = abs(data1[i,0]-data2[i,0])
        dy = abs(data1[i,1]-data2[i,1])
        diff += math.sqrt(dx*dx+dy*dy)
    return diff

def loss_center(data1,data2):
    #if data1.shape[0]!=data1.shape[0]:
    size = data1.shape[0]
    return loss(data1[int(size/4):int(3*size/4)],data2[int(size/4):int(3*size/4)])

def plotme(data1,data2):
    plt.plot(data1[:,0],data1[:,1])
    plt.plot(data1[:,0],data1[:,1],'bo')
    plt.plot(data2[:,0],data2[:,1])
    plt.plot(data2[:,0],data2[:,1],'ro')
    plt.title(f'Loss: {loss(data1,data2)}')

def plotmeX(data1,data2):
    
    plt.plot(data1[:,0],)
    plt.plot(data1[:,0],'bo')
    plt.plot(data2[:,0])
    plt.plot(data2[:,0],'ro')
    plt.title(f'Loss: {loss(data1,data2)}')


def load_file(filename):
    data = np.genfromtxt(filename,dtype=float,delimiter=',',skip_header=1)
    data = np.delete(data,np.where(data[:,3]!=0),axis=0)
    data = np.delete(data,np.where(data[:,1]>20),axis=0)
    data = np.delete(data,np.where(data[:,1]<-20),axis=0)
    data = np.delete(data,np.where(data[:,2]>20),axis=0)
    data = np.delete(data,np.where(data[:,2]<-20),axis=0)
    data = data[:,1:3]
    return data
    
    
##############################################

def find_sac(data,threshold=0.05):
    n = data.shape[0]
    starts = []
    lengths = []
    types = []
    current = -1
    i = 0
    while i<n-1:
        if data[i]>threshold: ## saccade
            types.append(1)
            starts.append(i)
            clen = 0
            while data[i]>threshold and i<n-1:
                clen = clen + 1
                i = i + 1
            lengths.append(clen)
            #print(starts[-1],lengths[-1],types[-1])
        else:    ## fixation
            types.append(0)
            starts.append(i)
            clen = 0
            while data[i]<=threshold and i<n-1:
                clen = clen + 1
                i = i + 1
            lengths.append(clen)
            #print(starts[-1],lengths[-1],types[-1])
    return starts,lengths,types

def make_signal(data,starts,lengths,types):
    fs = []
    #len = data.shape[0]
    i = 0
    for i in range(len(starts)):
        for j in range(lengths[i]):
            fs.append(types[i]) 
    return fs

def discard_short_fix(starts,lengths,types,short=40):
    i = 0
    while i < len(starts)-1:
        if types[i]==0 and lengths[i]<short:
            #print("i-1",starts[i-1],lengths[i-1],types[i-1])
            #print("i",starts[i],lengths[i],types[i])
            #print("i+1",starts[i+1],lengths[i+1],types[i+1])
            lengths[i-1] = lengths[i-1] + lengths[i] + lengths[i+1]
            del(lengths[i])
            del(lengths[i])
            del(starts[i])
            del(starts[i])
            del(types[i])
            del(types[i])
            #print("after",starts[i-1],lengths[i-1],types[i-1])
        else:    
            i = i + 1
    return starts,lengths,types


def calc_vel(data):
    n = data.shape[0]
    diff = np.zeros((n,2))
    #print(diff.shape)
    for i in range(n-1):
        diff[i,0] = data[i+1,0]-data[i,0]
        diff[i,1] = data[i+1,1]-data[i,1]
    v = np.sqrt(diff[:,0]**2,diff[:,1]**2)
    return v

# wyrównuje długość do wielokrotności d
def norm_length(x,d=4):
    if x%d!=0: x = x + (d- x%d)
    return x

def find_fix_sac(data,starts,lengths,types,factor):
    fixations = []
    saccades = []
    for i in range(len(starts)):
        s,l,t = starts[i],lengths[i],types[i]
    #for s,l,t in starts,lengths,types:
        if t==0:
            #print("F",l,'>',norm_length(l,factor))
            fixations.append(data[s:s+norm_length(l,factor)])
        else:
            #print("S",l,'>',norm_length(l,factor))
            saccades.append(data[s:s+norm_length(l,factor)])
    return fixations,saccades

def prepare_fix_sac(data,too_short_fix_dur=40,round_factor=32):
    v = calc_vel(data)
    v = np.array(v)
    starts,lengths,types = find_sac(v)    
    starts2,lengths2,types2 = discard_short_fix(starts.copy(),lengths.copy(),types.copy(),too_short_fix_dur)
    #s = make_signal(v,starts2,lengths2,types2)
    fixations,saccades = find_fix_sac(data,starts2,lengths2,types2,round_factor)
    return fixations,saccades

def calc_avg_fix_sac_dur(fixations,saccades):
    l=0
    for f in fixations:
        l = l + f.shape[0]
        #print("fshape",f.shape[0])
    print("fix len",l/len(fixations))
    fl = l/len(fixations)
    l=0
    for f in saccades:
        l = l + f.shape[0]
        #print("sshape",f.shape[0])
    print("sac len",l/len(fixations))
    sl = l/len(fixations)
    return fl,sl

def prepare_dataset(datain,FACTOR=8):
    samples = []
    for i in range(len(fixations)):
        samples.append(downsample(fixations[i],FACTOR))
    return samples,datain

######################################

def median_filter(data,kernel_size=3):
    temp = []
    kernel_size = kernel_size//2
    #print('ks=',kernel_size)
    data_final = np.zeros(len(data))
    for i in range(len(data)):
        temp = []
        for z in range(-kernel_size,kernel_size+1):
            #print(i,'>',i+z)
            if i+z>=0 and i+z<=len(data)-1:
                temp.append(data[i+z])
        temp.sort()
        data_final[i] = temp[len(temp)//2]
    return data_final

def median_f(data,kernel_size):
    new_data = []
    for i in range(len(data)):
        #print(data[i].shape)
        #print(data[i][:,0].shape)
        
        ndx = median_filter(data[i][:,0],kernel_size)
        ndy = median_filter(data[i][:,1],kernel_size)
        #print(ndx.shape)
        #print(ndy.shape)
        nd = np.vstack((ndx,ndy))
        #print(nd.shape)
        new_data.append(nd.T)
    return new_data

# d = [1,2,3,4,5,6,7,8]
# print(d)
# d2 = median_filter(d,7)
# print(d2)


######

def create_datasets(datain,factor=8,durations=[64]):
    ds = {}
    dl = {}
    for MAX in durations:
        trainSamples = []
        trainLabels = []
        for sc in datain:
            if sc.shape[0]>=MAX:
                trainSamples.append(downsample(sc[:MAX],factor))
                trainLabels.append(sc[:MAX])
        trainSamples = np.array(trainSamples)
        trainLabels = np.array(trainLabels)
        #print("max=",MAX,trainSamples.shape)
        #print("max=",MAX,trainLabels.shape)
        ds[MAX] = trainSamples
        dl[MAX] = trainLabels
    return ds,dl


from collections import Counter
def check_durations(data):
    x = []
    for i in range(len(data)):
        x.append(len(data[i]))
    print(Counter(x))

