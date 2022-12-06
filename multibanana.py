# For now this is a binary classifier (Unripe vs Ripe), heavily used the TensorFlow and Keras documentation and start-up guide
# Dependencies
# pip install tensorflow tensorflow-gpu opencv-python matplotlib
import tensorflow as tf
import numpy as np
import os
import cv2

from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential # consider functional for multiple outputs
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# our data pipeline
def process_data():
    # I don't wanna overkill my laptop so gonna do 8 images at a time
    data = tf.keras.utils.image_dataset_from_directory('data/multi')
    data = data.map(lambda x,y: (x/255, y)) # we need to scale our numbers down (currently 0 - 255)

    # partitioning according to general amount 80 / 10 / 10
    trainSize = int(len(data) * .8) # 80% training
    testSize = int(len(data) * .1) 
    validSize = int(len(data) * .1)
    
    # splitting data good doc here: https://towardsdatascience.com/how-to-split-a-tensorflow-dataset-into-train-validation-and-test-sets-526c8dd29438
    train = data.take(trainSize)
    test = data.skip(trainSize).take(testSize)
    valid = data.skip(trainSize + testSize).take(validSize)

    dataPack = [train, test, valid]
    return dataPack

# For now, we will use Sigmoid to determine unripe vs ripe, later we may switch to SoftMax for multi-stage classification
def create_cnn():
    model = Sequential()
    # We run multiple filters (conv2D first parameter) to detect multiple patterns (edges, textures, etc.)
    # Over time these filters can get more complicated and learn to detect complicated patterns (stems, etc.)
    model.add(Conv2D(2, (3,3), padding="same", activation='relu', input_shape=(256,256,3))) # 16 filters of 3 x 3, stride of 1
    # We pool our images so that the highlighted features (they will have the greatest value) are extracted while reducing the size of the image in the process
    model.add(MaxPooling2D())

    # These filters are randomly generated, so they find the patterns on their own
    model.add(Conv2D(4, (3,3), padding="same", activation='relu'))
    model.add(MaxPooling2D())
    #model.add(Dropout(0.2))

    # The reason we do multiple conv layers is because each conv layer can build a pattern
    # The first layer may find lines, the second layer may put those lines together to form a shape, etc.
   # model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
   # model.add(MaxPooling2D())

    # Our output from above is in the form of a matrix, which the dense layer cannot accept, so we use flatten to turn it into a vector that is acceptable
    model.add(Flatten())

    # The dense layer accepts input from ALL of the weights before it (hence 'dense'), this is used to classify images based off of the conv layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax')) # unripe, ripe, overripe

    # We need to use a different loss if we decide to go with softmax
    model.compile('adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy']) # adam is an optimizer for our model
    
    return model
    
def train_cnn(model, data):
    # Train
    train, test, valid = data

    tensorCallback = tf.keras.callbacks.TensorBoard(log_dir='train/multi/logs')
    # A major issue was that our model was overfitting, I've decided to lower the epoch amount and that seems to have fixed our issue
    network = model.fit(train, epochs=10, validation_data=valid, callbacks=[tensorCallback])
    
    # Test
    prec = Precision()
    recall = Recall()
    accur = BinaryAccuracy()
    
    
    test_loss, test_acc = model.evaluate(test, verbose=2)
    print("Accuracy: ", test_acc)
    plt.show()
    # Testing the model on external images
    img1 = cv2.imread('imgtest/b1.png')
    img2 = cv2.imread('imgtest/b2.jpg')
    img3 = cv2.imread('imgtest/b3.jpg')
    img4 = cv2.imread('imgtest/b4.jpg')
    img5 = cv2.imread('imgtest/b5.jpg')

    resize1 = tf.image.resize(img1, (256, 256))
    yhat1 = model.predict(np.expand_dims(resize1/255,0))
    resize2 = tf.image.resize(img2, (256, 256))
    yhat2 = model.predict(np.expand_dims(resize2/255,0))
    resize3 = tf.image.resize(img3, (256, 256))
    yhat3 = model.predict(np.expand_dims(resize3/255,0))
    resize4 = tf.image.resize(img4, (256, 256))
    yhat4 = model.predict(np.expand_dims(resize4/255,0))
    resize5 = tf.image.resize(img5, (256, 256))
    yhat5 = model.predict(np.expand_dims(resize5/255,0))

    yhatList = [yhat1, yhat2, yhat3, yhat4, yhat5]
    for index in range(5):
        test = yhatList[index].tolist()
        overripe, ripe, unripe = test[0]

        if overripe == max(test[0]):
            print("Overripe")
        elif ripe == max(test[0]):
            print("Ripe")
        else:
            print("Unripe")

    # Per the documentation, we can plot the models performance with this code
    fig = plt.figure()
    plt.plot(network.history['loss'], color='teal', label='loss')
    plt.plot(network.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(network.history['accuracy'], label='accuracy')
    plt.plot(network.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    return network, model

def init():
    # prevent TensorFlow from using all VRAM in GPU (Out Of Memory errors)
    #gpuList = tf.config.experimental.list_physical_devices('GPU')
    #for gpu in gpuList:
        #tf.config.experimental.set_memory_growth(gpu, True)
    pass

#init()
data = process_data()
model = create_cnn()
network, model = train_cnn(model, data)

# saving the model for later
model.save(os.path.join('models','multibanana.h5'))

# img4 = cv2.imread('test.png')
# resize4 = tf.image.resize(img4, (256, 256))

# newModel = load_model(os.path.join('models','binarybanana.h5'))
# yhat4 = newModel.predict(np.expand_dims(resize4/255,0))

# if yhat4 > 0.5:
#     print("The banana is UNRIPE")
# else:
#     print("The banana is RIPE")



