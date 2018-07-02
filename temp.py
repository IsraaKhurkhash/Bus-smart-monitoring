#import libraries from keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialising the CNN
classifaier=Sequential()

#convolution step
#here we select specific features from our image using feature detector,and we get many feature maps
classifaier.add(Convolution2D(32, 3, 3, input_shape=(100, 100, 3), activation='relu'))

#MaxPooling step to reduce the size of the feature maps,
#as result reduce the number of nodes in the (flatten step)
classifaier.add(MaxPooling2D(pool_size=(2, 2)))

#adding a second convolutional layer(applied not on images but on the feature maps coming from first conv layer)
classifaier.add(Convolution2D(32, 3, 3,  activation='relu'))
classifaier.add(MaxPooling2D(pool_size=(2, 2)))
#adding a third convolutional layer
#classifaier.add(Convolution2D(64, 3, 3,  activation='relu'))
#classifaier.add(MaxPooling2D(pool_size=(2, 2)))

#Flattening step to take all pooled feature maps and put them into one single vector 
classifaier.add(Flatten())

#Fully connection step 
#hidden layer
classifaier.add(Dense(output_dim=128, activation='relu' ))
#o/p layer
classifaier.add(Dense(output_dim=1, activation='sigmoid' ))

#compiling the CNN
classifaier.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting the cnn
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

    #applying image augmentation
train= train_datagen.flow_from_directory('resized(RG)/train',
                                         target_size=(100, 100),
                                         batch_size=32,
                                         class_mode='binary')
  
test = test_datagen.flow_from_directory('resized(RG)/test',
                                        target_size=(100, 100),
                                        batch_size=32,
                                        class_mode='binary')

classifaier.fit_generator(train,
                          steps_per_epoch=116,
                          epochs=25,
                          validation_data=test,
                          validation_steps=48)
 
#Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('resized(RG)/newprediction/upnor60.jpg', target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
prediction = 'normal'
else:
prediction = 'upnormal'