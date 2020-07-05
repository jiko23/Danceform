import os
import glob
import numpy as np
import pandas as pd
import skimage.io as io
from skimage.color import rgb2gray
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import load_model

"""
	Reading the test data and converting into grayscale and resizing. Resized 
	test data are stored into numpy array called x_test.
	Also reading the target from train data and labelizing them.
"""
train_df = pd.read_csv(r'E:\prog\hackerearth\0664343c9a8f11ea\dataset\train.csv')
test_df = pd.read_csv(r'E:\prog\hackerearth\0664343c9a8f11ea\dataset\test.csv')
test_img_dir = "E:/prog/hackerearth/0664343c9a8f11ea/dataset/test"

all_images = []

for i in range(0,len(test_df)) :

	test_data_path = os.path.join(test_img_dir,test_df['Image'].loc[i])
	test_files = glob.glob(test_data_path)

	for f1 in test_files:
		img = io.imread(f1)
		img1 = rgb2gray(img)
		img1 = img1.reshape([200, 150,1])
		#print(img1.shape)
		all_images.append(img1)

x_test = np.array(all_images)
del all_images[:]

target_ = LabelBinarizer()
target_.fit(train_df['target'])
y_train = target_.transform(train_df['target'])

########################################## Predictions #################################################################################
model_tf = load_model(r'E:\prog\hackerearth\0664343c9a8f11ea\dataset\image_model.hdf5') ## LOADING THE TRAINED MODEL FROM PATH

_labels = target_.classes_

pred_list = []
for i in range(0,len(x_test)) :
	prediction = model_tf.predict([np.array([x_test[i]])])
	predicted_label= _labels[np.argmax(prediction[0])]
	pred_list.append([test_df['Image'].iloc[i],predicted_label])

classify = pd.DataFrame(pred_list,columns=['Image','Prediction'])
classify.to_csv(r'E:\prog\hackerearth\0664343c9a8f11ea\dataset\danceform_pred.csv') ## saving the result dataframe into .csv format
#########################################################################################################################################