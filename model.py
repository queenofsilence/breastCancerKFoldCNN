from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from importingData import save_dictionary
import tensorflow as tf

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

    def TrainModel(self, model, x_train, y_train, epochs, counter):
        history = model.fit(x_train, y_train,epochs=epochs, batch_size=32)
        save_dictionary('histories\history'+counter+'.dat', history.history)