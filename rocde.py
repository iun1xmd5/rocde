
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 06 02:27:08 2021

@author: c1ph3r
"""
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Model
from keras.layers.merge import concatenate
import pandas as pd
import numpy as np
import os
from keras.utils import plot_model
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
import tqdm
from tweek import set_size
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
#plt.style.use('mpsty/mypaperstyle.mplstyle')
#mpl.use('pdf')
#import matplotlib.pyplot as plt
#from matplotlib import style
#style.use('seaborn-paper')
'''plt.rc('font', family= 'serif', serif= 'Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman')
#plt.rc('font', family='sans-serif', sans_serif= 'Helvetica, Avant Garde, Computer Modern Sans serif')
plt.rc('font', family='sans-serif')
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('font', family='monospace', monospace= 'Courier, Computer Modern Typewriter')'''
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('text', usetex = True) # Use latex for text
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 14)
plt.rc('axes', linewidth=1)
plt.rcParams['text.latex.preamble'] =[r'\boldmath']
params = {'legend.framealpha': 0.1,
          'legend.handlelength': 0.8,
          'legend.labelspacing':0.1,
          'legend.fontsize' : 10}
plt.rcParams.update(params)

class Convnet(): 
    '''
    Hybrid parallel convolutional neural network model with three Channels, 
    whose output are merged through concatenation.
    
    Parameters
    ----------
    batch_size: int
            the size of the batch
    epochs: int
            number of epochs to be conducted during training
    height: int
            the size of the height (h) of the filter
    width: int
            the size of the width (w) of the filter
            
   
    Returns
   
    -------
    ftted model and predictions: tuple
            The fitted model and predictions for road crack detection
    '''

    def __init__(self,X,y,sr,batch_size=1000,epochs=3,h=50,w=50,c=3):
        self.X=X
        self.y =y
        self.sr =sr
        self.batch_size=batch_size
        self.epochs=epochs
        self.c=c
        self.w=w
        self.h=h
        self.X_train,self.y_train = self.X[:sr,:],self.y[:sr]

    def pcnn(self):
        self.input1 = Input(shape=(self.h,self.w,self.c))
        conv1 = Conv2D(64,(4,4),activation='relu')(self.input1)
        drop1 =Dropout(0.3)(conv1)
        pool1 = MaxPooling2D(2,2)(drop1)
        flat1 = Flatten()(pool1)
        #input channel 2
        self.inputs2 = Input(shape=(self.h,self.w,self.c))
        conv2= Conv2D(filters=40, kernel_size=(3,3),
                                      activation='relu')(self.inputs2)
        drop2 =Dropout(0.3)(conv2)
        pool2 = MaxPooling2D(pool_size=(2,2))(drop2)
        flat2 = Flatten()(pool2)
        #input channel 3
        self.inputs3 = Input(shape=(self.h,self.w,self.c))
        conv3= Conv2D(filters=32, kernel_size=(2,2),
                                      activation='relu')(self.inputs3)
        drop3 = Dropout(0.3)(conv3)
        #conv3 = TimeDistributed(Conv1D(filters=16, kernel_size=1, activation='relu'))(conv3)
        pool3 = MaxPooling2D(pool_size=(2,2))(drop3)
        flat3 = Flatten()(pool3)
        #merged
        merged =concatenate([flat1,flat2, flat3])
        self.merged=merged

    def fit(self):
        self.pcnn()
        dense1 = Dense(50,activation='relu')(self.merged)
        self.out= Dense(2,activation='sigmoid')(dense1)
        model = Model([self.input1,self.inputs2,self.inputs3],self.out)
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        history=model.fit([self.X_train,self.X_train,self.X_train],self.y_train,
                  epochs=self.epochs,batch_size=self.batch_size,verbose=1)
        self.model=model
        self.history = history

    def predict(self,X,y):
        self.ypred=self.predict(X, y)
