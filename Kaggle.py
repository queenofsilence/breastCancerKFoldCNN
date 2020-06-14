from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
import tensorflow as tf
import cv2

no_angles = 360
url = '/kaggle/input/mias-mammography/all-mias/'

def save_dictionary(path,data):
        print('saving catalog...')
        #open('u.item', encoding="utf-8")
        import json
        with open(path,'w') as outfile:
            json.dump(str(data), fp=outfile)
        # save to file:
        print(' catalog saved')

def read_image():
    import cv2
    info = {}
    for i in range(322):
        if i<9:
            image_name='mdb00'+str(i+1)
        elif i<99:
            image_name='mdb0'+str(i+1)
        else:
            image_name = 'mdb' + str(i+1)
        # print(image_name)
        image_address= url+image_name+'.pgm'
        img = cv2.imread(image_address, 0)
        # print(i)
        img = cv2.resize(img, (64,64))   #resize image

        rows, cols = img.shape
        info[image_name]={}
        for angle in range(no_angles):
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)    #Rotate 0 degree
            img_rotated = cv2.warpAffine(img, M, (cols, rows))
            info[image_name][angle]=img_rotated
    return (info)

import os
import sys

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))    

def read_lable():
    filename = url+'Info.txt'
    text_all = open(filename).read()
    #print(text_all)
    lines=text_all.split('\n')
    info={}
    for line in lines:
        words=line.split(' ')
        if len(words)>3:
            if (words[3] == 'B'):
                info[words[0]] = {}
                for angle in range(no_angles):
                    info[words[0]][angle] = 0
            if (words[3] == 'M'):
                info[words[0]] = {}
                for  angle in range(no_angles):
                    info[words[0]][angle] = 1
    return (info)

class Model:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def createModel(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(self.rows, self.cols, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        #model.add(Activation('softmax'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        #model.add(Activation('tanh'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Activation('softmax'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(
    num_thresholds=200,
    curve="ROC",
    summation_method="interpolation",
    name=None,
    dtype=None,
    thresholds=None,
    multi_label=False,
    label_weights=None), tf.keras.metrics.Recall(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
), tf.keras.metrics.Precision(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
),])
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Accuracy()])#,    tf.keras.metrics.CosineSimilarity(), tf.keras.metrics.LogCoshError()])
        return model

    def TrainModel(self, model, x_train, y_train, epochs):
        history = model.fit(x_train, y_train,epochs=epochs, batch_size=32)
        save_dictionary('history1.dat', history.history)
        
from sklearn.model_selection import KFold
import numpy as np

class Data:
    def __init__(self):
        self.no_angles = 360 # 360°
        self.X = np.array([])
        self.Y = np.array([])

    def SplitingData(self):
        from sklearn.model_selection import train_test_split
        import numpy as np
        lable_info=read_lable()
        image_info=read_image()
        #print(image_info[1][0])
        ids=lable_info.keys()   #ids = acceptable labeled ids
        #print(type(ids))
        del lable_info['Truth-Data:']       
        #print(lable_info)
        #print(ids)
        X=[]
        Y=[]
        for id in ids:
            for angle in range(self.no_angles):
                X.append(image_info[id][angle])
                Y.append(lable_info[id][angle])
        self.X=np.array(X)
        self.Y=np.array(Y)
        return [X,Y]

    def kfold_Split(self, n_split, epoch):
        for train_index,test_index in KFold(n_split).split(self.X):
            x_train,x_test = self.X[train_index],self.X[test_index]
            y_train,y_test = self.Y[train_index],self.Y[test_index]
            rows, cols = x_train[0].shape
            (a, b, c) = x_train.shape
            x_train = np.reshape(x_train, (a, b, c, 1))
            (a, b, c) = x_test.shape
            x_test = np.reshape(x_test, (a, b, c, 1))
            first = Model(rows, cols)
            model = first.createModel()
            print("Begin Training...")
            # Train Model using 
            first.TrainModel(model, x_train, y_train, epoch) # n epoch
            print('Model evaluation ', model.evaluate(x_test,y_test))
            

def main():
   data = Data()
   X,Y = data.SplitingData()
   data.kfold_Split(2, 1) # first number is split number second one is the number of epochs => kfold_split(split, epoch)
main()        