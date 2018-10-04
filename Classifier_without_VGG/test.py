import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.preprocessing import image

model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

images_test = ['Dataset/single_prediction/cat_or_dog_2.jpeg', 'Dataset/single_prediction/cat_or_dog_1.jpg', 'Dataset/single_prediction/bus.jpg', 'Dataset/single_prediction/flower.jpeg', 'Dataset/single_prediction/person2.jpeg', 'Dataset/single_prediction/person3.jpeg', 'Dataset/single_prediction/airplane_0715.jpg', 'Dataset/single_prediction/flower_0838.jpg', 'Dataset/single_prediction/flower_0839.jpg', 'Dataset/single_prediction/motorbike_0732.jpg']

for image_ind in images_test:
	test_image = image.load_img(image_ind, target_size = (64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	array = model.predict(test_image)
	answer = np.argmax(array, axis = 1)
	print(image_ind + ' : ' + str(array))
	if answer == 0:
		prediction = 'airplane'
	elif answer == 1:
		prediction = 'car'
	elif answer == 2:
		prediction = 'cat'
	elif answer == 3:
		prediction = 'dog'
	elif answer == 4:
		prediction = 'flower'
	elif answer == 5:
		prediction = 'fruit'
	elif answer == 6:
		prediction = 'motorbike'
	else:
		prediction = 'person'
	print(prediction)
