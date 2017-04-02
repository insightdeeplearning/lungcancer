from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
from keras import backend as K
from sklearn.model_selection import train_test_split
import cv2, numpy as np

"""
Modified VGG_16 architecture for 3D image classifier. 
Requirement: 
The input image must be greater than 224*224 and it has to be a color image
such as length*width*3. The formate of input_shape = z, l, w, color_channel. 
"""
def VGG_16(weights_path=None):
    # print batch_size, image_shape, num_classes
    model = Sequential()
    model.add(ZeroPadding3D((1,1,1),input_shape=(224,224,224,3)))
    
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(128, (3, 3, 3), activation='relu'))
    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(128, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(256, (3, 3, 3), activation='relu'))
    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(256, (3, 3, 3), activation='relu'))
    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(256, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(512, (3, 3, 3), activation='relu'))
    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(512, (3, 3, 3), activation='relu'))
    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(512, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(512, (3, 3, 3), activation='relu'))
    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(512, (3, 3, 3), activation='relu'))
    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(512, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

    model.add(Flatten()) # flatten previous layer into single vector
    model.add(Dense(4096, activation='relu')) # FC layer
    model.add(Dropout(0.5)) 
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))#2 denotes two possible outcomes

    # if a weights path is supplied (inicating that the model was pre-trained), then load the weights
    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    X_train = np.load('simulated_X_train-numpoint20-z224-l224-w224-c3.npy')
    Y_train = np.load('simulated_Y_train_numpoint20.npy')
    # X_train.shape = (20,8,224,224,3)
    # print "construct the model"
    model = VGG_16()
    #  # For a binary classification problem
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, nb_epoch=1, batch_size=1, verbose=1)
    
    # save your model
    from keras.models import load_model
    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model
    # returns a compiled model
    # identical to the previous one
    model = load_model('my_model.h5')

    """fit the test data to the model"""
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # print('Test accuracy:', score[1])

