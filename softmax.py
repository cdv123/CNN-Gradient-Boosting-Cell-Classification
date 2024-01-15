import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, AveragePooling2D, Input
from keras import Sequential, optimizers, layers
from keras.models import load_model
from imblearn.over_sampling import RandomOverSampler
import keras_tuner as kt
from keras import backend as K
from scikeras.wrappers import KerasClassifier
import random

class ImageDataset:
    def __init__(self, images, labels):
        self.images = images/255
        self.labels = labels
        self.class_num = len(np.unique(labels))
        self.counts = []
        self.proportions = []
        self.length = np.shape(images)[0]
        self.width = np.shape(images)[1]
        self.one_hot_labels = self.one_hot_encode()
        self.update_counts()
        # self.oversample()

    def update_counts(self):
        self.counts = []
        self.proportions = []
        
        for i in range(self.class_num):
            self.counts.append(len(np.where(self.labels == i)[0]))
        
        self.proportions = [count/self.length for count in self.counts]

    def oversample(self):
        ros = RandomOverSampler(random_state=0)
        self.images= self.images.reshape((self.length, self.width*self.width*3))
        self.images, self.labels = ros.fit_resample(self.images, self.labels)
        self.length = self.images.shape[0]
        self.images = self.images.reshape((self.length, self.width, self.width, 3))
        self.one_hot_labels = self.one_hot_encode()
        self.update_counts()
        self.image_features = self.images

    def one_hot_encode(self):
        one_hot_labels = np.array([np.zeros(self.class_num) for i in range(self.length)])
        for i in range(self.length):
            one_hot_labels[i][self.labels[i]] = 1
        return one_hot_labels
    
    # def apply_CNN(self, model):

    def shuffle(self):
        p = np.random.permutation(self.length)
        self.images, self.labels, self.one_hot_labels = self.images[p], self.labels[p], self.one_hot_labels[p]

    def get_features(self, model):
        print(np.shape(self.images), np.shape(self.one_hot_labels))
        self.image_features = model.predict(self.images)

# model.add(BatchNormalization())

class CNN(kt.HyperModel):

    def build(self, hp):
        K.clear_session()
        # define hyperpamaters search space for tuning
        filters_1 = hp.Int('filters_1', min_value=16, max_value=250, step=32)
        filters_2 = hp.Int('filters_2', min_value=16, max_value=500, step=32)
        filters_3 = hp.Int('filters_3', min_value=16, max_value=500, step=32)
        dense_1_size = hp.Int('size_1', min_value=10, max_value=200, step=10)
        learning_rate = hp.Choice('learning_rate', values = [1e-2, 5e-3, 1e-3, 1e-4, 1e-5])

        model = Sequential()
        model.add(Input(shape = (28, 28, 3)))
        model.add(Conv2D(filters=filters_1, kernel_size=(3, 3), activation='relu', input_shape = (28, 28, 3), strides=1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(filters=filters_2, kernel_size=(3, 3), activation='relu', strides=1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(filters = filters_3, kernel_size=(3, 3), activation='relu', strides=1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Flatten())
        model.add(Dropout(.1))
        model.add(Dense(dense_1_size, activation='relu'))
        model.add(Dense(8, activation='softmax'))
        optimizer = optimizers.SGD(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=8,
            verbose=0,
            epochs=11,
            **kwargs
        )


data = np.load("bloodmnist.npz")
train_images = data["train_images"]
print(np.shape(data["train_images"]))
val_images = data["val_images"]
print(np.shape(data["val_images"]))
test_images = data["test_images"]
print(np.shape(data["test_images"]))
train_labels = data["train_labels"]
print(np.shape(data["train_labels"]))
val_labels = data["val_labels"]
print(np.shape(data["val_labels"]))
test_labels = data["test_labels"]
print(np.shape(data["test_labels"]))

random.seed(0)
np.random.seed(0)

# Initialise Class for training, validation, test
train_dataset = ImageDataset(train_images, train_labels)
val_dataset = ImageDataset(val_images, val_labels)
test_dataset = ImageDataset(test_images, test_labels)

# print counts and proportions to see if data needs to be balanced
print(train_dataset.counts, train_dataset.proportions)
print(val_dataset.counts, val_dataset.proportions)
print(test_dataset.counts, test_dataset.proportions)

train_dataset.oversample()
# print(train_dataset.counts)
# print(np.shape(train_dataset.images))
# print(np.shape(train_dataset.one_hot_labels))
# one hot encode labels

# def one_hot_encode(label):
#     return utils.to_categorical(label, num_classes=train_dataset.class_num)
# one_hot_train = np.array([utils.to_categorical(label, num_classes=train_dataset.class_num) for label in train_dataset.labels])
# print(np.shape(one_hot_train))
# one_hot_train = one_hot_train.reshape(np.shape(one_hot_train)[0], train_dataset.class_num)
# print(np.shape(one_hot_train))

train_dataset.shuffle()

def add_preprocessing(model):
    # model.add(layers.RandomBrightness(factor=0.2))
    model.add(layers.RandomFlip(mode="horizontal_and_vertical"))
    model.add(layers.RandomZoom(height_factor=0.2))
    # model.add(layers.RandomRotation(factor=0.2))
    model.add(layers.RandomContrast(factor=0.2))
    return model

tuner = kt.RandomSearch(
    CNN(),
    objective='val_accuracy',
    directory='optimal_parameters',
    project_name='train_CNN'
)

tuner.search(train_dataset.images, train_dataset.one_hot_labels, validation_data=(val_dataset.images, val_dataset.one_hot_labels))

# model = Sequential()
# model.add(Input(shape = (28, 28, 3)))
# model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu', input_shape = (28, 28, 3), strides=1))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Conv2D(filters=112, kernel_size=(3, 3), activation='relu', strides=1))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Conv2D(filters = 112, kernel_size=(3, 3), activation='relu', strides=1))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Flatten())
# model.add(Dropout(.1))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(8, activation='softmax'))
# optimizer = optimizers.SGD(learning_rate=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# model.fit(train_dataset.images, train_dataset.one_hot_labels, epochs=11, validation_data=(val_dataset.images, val_dataset.one_hot_labels), batch_size=16)