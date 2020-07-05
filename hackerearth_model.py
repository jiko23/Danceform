import os
import glob
import pandas as pd
import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
import keras
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint


"""
	Reading training datasets
"""
train_df = pd.read_csv(r'E:\prog\hackerearth\0664343c9a8f11ea\dataset\train.csv')

"""
	setting training image folder paths.
	Reading images from training and testing folders.
	Converting images into grayscale and resizing images.
	Storing images into numpy arrays i.e. training images
	are stored into x_train. Labelizing image targets/categories
	and storing into y_train.
"""
train_img_dir = "E:/prog/hackerearth/0664343c9a8f11ea/dataset/train"

all_images = []
for i in range(0,len(train_df)) :

	train_data_path = os.path.join(train_img_dir,train_df['Image'].loc[i])
	train_files = glob.glob(train_data_path)

	for f1 in train_files:
		img = io.imread(f1)
		img1 = rgb2gray(img)
		img1 = img1.reshape([200, 150,1])
		#print(img1.shape)
		all_images.append(img1)

x_train = np.array(all_images)
del all_images[:]


target_ = LabelBinarizer()
target_.fit(train_df['target'])
y_train = target_.transform(train_df['target'])

###################################### Model_ ##########################################
"""
	Buiding keras sequental model. Traing the model with train data.
	Saving the trained model.
"""
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=[200, 150, 1]))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(y_train.shape[1], activation="softmax"))

model.summary()

model.compile(loss=keras.losses.BinaryCrossentropy(),optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

filepath= r"E:\prog\hackerearth\0664343c9a8f11ea\dataset\image_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') ### saving the trained model
callbacks_list = [checkpoint]

model.fit(x_train, y_train,validation_split=0.20, batch_size=20, callbacks=callbacks_list,epochs=200,shuffle=True, verbose=1)

train_score = model.evaluate(x_train,y_train,batch_size=20,verbose=1) ## TRAINING EVALUATION ALONG WITH LOSS AND ACCURACY SCORE
print('Train[TRAIN_LOSS,ACCURACY]:', train_score) ##### train set loss and accuracy

###########################################################################################################################################