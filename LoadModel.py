from keras.models import load_model
model = load_model('classification-keras.h5')

#Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(test_data_dir + '/nor60.jpg', target_size = (img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = binaryclassifier.predict(test_image)

print(result)  
    