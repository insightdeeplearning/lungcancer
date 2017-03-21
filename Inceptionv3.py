# Assuming we are using Tensorflow backend without GPU Support

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from FetchImages import Fetch_Melanoma_Data
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras.layers import Input
import numpy as np

# grab the  dataset 
print("[INFO] Loading Images...")
width = 299
height = 299
depth = 3
DataPath = '/Users/staples/Documents/Transfer Learning/ImageDB/'
dataset = Fetch_Melanoma_Data(DataPath,width,height,depth)


data = dataset['images']
target = dataset['target']
target_names = dataset['target_names']
del dataset

(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, target.astype("float32"), test_size=0.33)

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; 
# trainLabels = np_utils.to_categorical(trainLabels, 2)
trainLabels = np_utils.to_categorical(trainLabels, 2)
testLabels = np_utils.to_categorical(testLabels, 2)

# initialize the optimizer and model
print("[INFO] Building InceptionV3 model - with weights loaded...")

# create the base pre-trained model
# Tensorflow expects the image data to be (y,x,z)
input_tensor = Input(shape=(height, width, depth))  

base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer with 2 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
    

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
	metrics=["accuracy",'recall'])
model.fit(trainData.transpose(0,2,3,1), trainLabels, batch_size=32, nb_epoch=5,verbose=1)

# # show the accuracy on the testing set
print("[INFO] evaluating...")
y_pred = model.predict(testData, batch_size=32, verbose=1)

(loss, accuracy, recall) = model.evaluate(testData.transpose(0,2,3,1), testLabels, batch_size=32, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
print("[INFO] Recall: {:.2f}%".format(recall * 100))