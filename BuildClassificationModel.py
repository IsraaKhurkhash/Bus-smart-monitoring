import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense



img_width, img_height = 150, 150
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
train_path = "/train"
testing_path = "/test"

train_data_dir =script_dir + train_path
test_data_dir = script_dir + testing_path

######
#image augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# used to rescale the pixel values from [0, 255] to [0, 1] interval
test_datagen = ImageDataGenerator(rescale=1./255)
######



# automagically retrieve images and their classes for train and test sets
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

#initialising the CNN
binaryclassifier = Sequential()

#convolution step
#here we select specific features from our image using feature detector,and we get many feature maps
binaryclassifier.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height,3)))
binaryclassifier.add(Activation('relu'))

#MaxPooling step to reduce the size of the feature maps,
#as result reduce the number of nodes in the (flatten step)
binaryclassifier.add(MaxPooling2D(pool_size=(2, 2)))

#adding a second convolutional layer
binaryclassifier.add(Convolution2D(32, 3, 3))
binaryclassifier.add(Activation('relu'))
binaryclassifier.add(MaxPooling2D(pool_size=(2, 2)))

#adding a third convolutional layer
binaryclassifier.add(Convolution2D(64, 3, 3))
binaryclassifier.add(Activation('relu'))
binaryclassifier.add(MaxPooling2D(pool_size=(2, 2)))

#Flattening step to take all pooled feature maps and put them into one single vector 
binaryclassifier.add(Flatten())
#Fully connection step 
#hidden layer
binaryclassifier.add(Dense(64))
binaryclassifier.add(Activation('relu'))
binaryclassifier.add(Dropout(0.5))
#o/p layer
binaryclassifier.add(Dense(1))
binaryclassifier.add(Activation('sigmoid'))

binaryclassifier.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#fitting the cnn
nb_epoch = 30
nb_train_samples = 116
nb_test_samples = 48

binaryclassifier.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=test_generator,
        nb_val_samples=nb_test_samples)


#Returns the loss value & metrics values for the model in test mode.
binaryclassifier.evaluate_generator(test_generator, nb_test_samples)

#saving the model
binaryclassifier.save('C:/Users/FALCON/Desktop/test/Two-class-classification-using-keras/classification-keras.h5')

#Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(test_data_dir + '/nor72.jpg', target_size = (img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = binaryclassifier.predict(test_image)


print(result)  
    
