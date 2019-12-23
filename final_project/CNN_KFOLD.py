import os
import numpy as np; np.random.seed(42);
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold
import cv2
from sklearn.metrics import confusion_matrix
import json

CV = 10
def add_class(filename):
    k = int(filename.split('__')[1])
    if k <= 282:
        return 0
    if k <= 287:
        return 1
    if k <= 291:
        return 2
    return 3    
    

images = []
classes = []
path = 'project/experiment/data/train/images'
ufs = os.listdir(path)
for uf in ufs:
    pathtemp = os.path.join(path, uf)
    for image in os.listdir(pathtemp):
        image = os.path.join(pathtemp, image)
        images.append(cv2.imread(image, 0))
        classes.append(add_class(image))
        
path = 'project/experiment/data/test/images'
ufs = os.listdir(path)
for uf in ufs:
    pathtemp = os.path.join(path, uf)
    for image in os.listdir(pathtemp):
        image = os.path.join(pathtemp, image)
        images.append(cv2.imread(image, 0))
        classes.append(add_class(image))
images = np.array(images)
images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
images = images.astype('float32')/255
classes = np.array(classes)
classes = np_utils.to_categorical(classes, 4)

kfold = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)
results = []

a = np.zeros(CV)
b = np.zeros(shape=(images.shape[0], 1))

for index_train, index_test in kfold.split(images, np.zeros(shape=(images.shape[0], 1))):
#    print('index train:', index_train);print('index test:', index_test);
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(16, 16), 
                     input_shape=(images.shape[1], images.shape[2], 1),
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(8, 8)))
    
    model.add(Flatten())
    
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(.3))
    
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(.3))
    
    model.add(Dense(units=4, activation='softmax'))
    
    filepath="NEW__weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    call_mc = ModelCheckpoint('callbacks_models/'+filepath, monitor='val_acc', 
                              save_best_only=True, mode='max', verbose=1)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    model.fit(images[index_train], classes[index_train], epochs=160)
    precision = model.evaluate(images[index_test], classes[index_test])
    results.append(precision)
