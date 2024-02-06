#	-	-	-	-	-	-	-	-	IMPORTS	-	-	-	-	-	-	-	-	#   
import numpy as np                                                          #
import pandas as pd                                                         #
import tensorflow as tf                                                     #
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout,\
                         BatchNormalization, DepthwiseConv2D, RandomCrop
                         
from keras.models import Sequential                                         #
from tensorflow.keras.preprocessing.image import ImageDataGenerator         #
#---------------------------------------------------------------------------#

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
train_set = train_datagen.flow_from_directory('celebdataset/train',         #
                                              target_size = img_size,       #
                                              batch_size = batch_size,      #
                                              class_mode = 'categorical')   #
#---------------------------------------------------------------------------#

#	-	-	-	-	-	-	-	-	MODEL PARAMS-	-	-	-	-	-	-	-	-	#
#100.00% EPOCH 6
#WILLOW
model = Sequential ([
    Conv2D(16, 4, activation='relu', input_shape=(img_scale,img_scale,3)),
    Dropout(rate=0.125),
    BatchNormalization(),
    MaxPooling2D((4,4)),
    Conv2D(8, 2, activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    
#-------------------------------------END MODEL-------------------------------------#

#	-	-	-	-DONT TOUCH-	-	-	-	-	-	#
    Flatten(),                                      #
    Dense(64, activation='relu'),  #Except this one #
    #Dropout(rate=0.125),             #and this one #
    Dense(num_cats, activation='softmax')           #
])                                                  #
                                                    #
#---------------------------------------------------#

#	-	-	-	-	-	-	-	-	FINAL PARAMS-	-	-	-	-	-	-	-	-	#
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_set, epochs=10)
model.save('CelebsModel.h5', include_optimizer=True)
model.evaluate(train_set)

# EX - EX - EX - EX - EX - EX - EX - EX - EX - EX - EX - EX - EX - EX - EX - EX #
exampleLayers = Sequential ([                                                   #
    #DATA AUGMENTATION                                                          #
                                                                                #
    #Random Crop: Crops random section of image of height x width               #
    RandomCrop(64,64),                                                          #
                                                                                #
    #typical cnn arch is alternation conv and pool, then dense                  #
    #Conv: Num filters, shape of filter, activation, (input layer shape)        #
    Conv2D(4, (2,2), activation='relu', input_shape=(img_scale,img_scale,3)),   #
                                                                                #    
    #pooling layer: size of pooling data                                        #
    MaxPooling2D((2,2)),                                                        #
                                                                                #
    #Depthwise Conv: shape of filter, activation                                #
    #This one Willow did research for, seems like a good alt to Conv2D that uses#
    #individual filters for each input channel, where Conv2D uses 1.            #
    DepthwiseConv2D((2, 2), activation='relu'),                                 #
                                                                                #
    #Dropout: Rate (% of drop out is 1/1-x, for x)                              #
    #Higher this value, more nodes are shut off per run. prevents over-fitting  #
    Dropout(rate=0.3),                                                          #
                                                                                #
    #Batch Normalization                                                        #
    BatchNormalization(),                                                       #
                                                                                #
    #Flatten                                                                    #
    Flatten(),                                                                  #
    #Dense                                                                      #
    Dense(8, activation='relu'),                                                #
                                                                                #
    #CNNs end if softmax. num_cats needs to be number of classifications.       #
    Dense(num_cats, activation='softmax')                                       #
])                                                                              #
# EX - EX - EX - EX - EX - EX - EX - EX - EX - EX - EX - EX - EX - EX - EX - EX #


#---------------------█░█ █ █▀▀ █░█ █▀ █▀▀ █▀█ █▀█ █▀▀ █▀---------------------#
#---------------------█▀█ █ █▄█ █▀█ ▄█ █▄▄ █▄█ █▀▄ ██▄ ▄█---------------------#
#100.00% EPOCH 6
#WILLOW
"""
model = Sequential ([
    Conv2D(16, 4, activation='relu', input_shape=(img_scale,img_scale,3)),
    Dropout(rate=0.125),
    BatchNormalization(),
    MaxPooling2D((4,4)),
    Conv2D(8, 2, activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
])

#100.00% EPOCH 10
#WILLOW
model = Sequential ([
    Conv2D(32, 4, activation='relu', input_shape=(img_scale,img_scale,3)),
    Dropout(rate=0.25),
    BatchNormalization(),
    MaxPooling2D((4,4)),
    Conv2D(8, 2, activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
])

#99.42% EPOCH 10
#WILLOW
model = Sequential ([
    Conv2D(64, 3, activation='relu', input_shape=(img_scale,img_scale,3)),
    Dropout(rate=0.333),
    BatchNormalization(),
    MaxPooling2D((4,4)),
    Conv2D(16, 2, activation='relu'),
    Dropout(rate=0.25),
    BatchNormalization(),
    MaxPooling2D((2,2)),
])

#98.43% EPOCH 10
#WILLOW
model = Sequential ([
    Conv2D(16, (3,3), activation='relu', input_shape=(img_scale,img_scale,3)),
    Dropout(rate=0.125),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(4, (2,2), activation='relu'),
    Dropout(rate=0.125),
    BatchNormalization(),
    MaxPooling2D((2,2)),
])

#97.60% EPOCH 10
#WILLOW
model = Sequential ([
    Conv2D(8, (3,3), activation='relu', input_shape=(img_scale,img_scale,3)),
    Dropout(rate=0.125),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(8, (2,2), activation='relu'),
    Dropout(rate=0.125),
    BatchNormalization(),
    MaxPooling2D((2,2)),
])

#95.53% EPOCH 10
#WILLOW
model = Sequential ([
    Conv2D(16, (8,8), activation='relu', input_shape=(img_scale,img_scale,3)),
    Dropout(rate=0.2),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(32, (4,4), activation='relu'),
    Dropout(rate=0.2),
    BatchNormalization(),
    MaxPooling2D((2,2)),
])

#94.21% EPOCH 10
#WILLOW
model = Sequential ([
    Conv2D(8, (8,8), activation='relu', input_shape=(img_scale,img_scale,3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(rate=0.5),
    Conv2D(32, (4,4), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
])

#87.17 EPOCH 10
#WILLOW
model = Sequential ([
    Conv2D(8, (8,8), activation='relu', input_shape=(img_scale,img_scale,3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(rate=0.5),
    Conv2D(32, (4,4), activation='linear'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
])
"""
