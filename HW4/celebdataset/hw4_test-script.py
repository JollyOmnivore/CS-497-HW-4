#	-	-	-	-	-	-	-	-	IMPORTS	-	-	-	-	-	-	-	-	#   
import numpy as np                                                          #
import pandas as pd                                                         #
import tensorflow as tf                                                     #
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout,\
                         BatchNormalization, DepthwiseConv2D, RandomCrop
                         
from keras.models import Sequential                                         #
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model#
#---------------------------------------------------------------------------#

model = load_model('CelebsModel.h5')

#	-	-HYPERPARAMS-	-	#
#   WE CAN PLAY WITH THESE  #
img_scale = 128
batch_size = 4
#---------------------------#

#	-	-	-	-	-	-	-	-DONT-TOUCH-	-	-	-	-	-	-	-	#
num_cats = 15                                                               #
img_size = (img_scale,img_scale)                                            #
train_datagen = ImageDataGenerator(rescale = 1.0/255)                       #
                                                                            #
test_set = train_datagen.flow_from_directory('celebdataset/test',         #
                                              target_size = img_size,       #
                                              batch_size = batch_size,      #
                                              class_mode = 'categorical')   #
#---------------------------------------------------------------------------#


#	-	-	-	-	-	-	-	-	FINAL PARAMS-	-	-	-	-	-	-	-	-	#

model.fit(test_set, epochs=10)
model.evaluate(test_set)
