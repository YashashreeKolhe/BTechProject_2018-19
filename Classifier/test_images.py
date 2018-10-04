from keras.applications.vgg16 import VGG16
from os import walk
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

model = VGG16()
print(model.summary())

picture_array = list()
test_path = 'single_prediction'

for (dirpath, dirnames, filenames) in walk(test_path):
    for filename in filenames:
    	picture_array.append(test_path + "/" + filename)
    break

for picture in picture_array:
	image = load_img(picture, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	yhat = model.predict(image)
	label = decode_predictions(yhat)
	label = label[0][0]
	print('%s %s (%.2f%%)' % (picture, label[1], label[2]*100))	
