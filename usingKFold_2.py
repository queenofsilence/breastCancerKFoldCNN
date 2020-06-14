from sklearn.model_selection import KFold
from importingData import read_image
from importingData import read_lable
from model import Model
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
