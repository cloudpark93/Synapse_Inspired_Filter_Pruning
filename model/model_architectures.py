from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, MaxPool2D, GlobalAveragePooling2D, Add, Input
from keras import regularizers
from tensorflow.keras import Model
import tensorflow as tf

def model_type(architecture):
    if architecture == 'vgg16':
        # https://github.com/aknakshay/VGG16---CIFAR10/blob/master/VGG16.ipynb
        # https://github.com/garyliu0816/Keras-VGG-CIFAR10/blob/master/keras-VGG16-cifar10.ipynb
        model = Sequential()
        weight_decay = 0.001

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(10))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # initially lr=1e-3, decay=5e-4

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        return model


    elif architecture == 'resnet56':
        # https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/resnet-cifar10-2.2.1.py
        n = 9  # 56 layers
        channels = [16, 32, 64]

        inputs = Input(shape=(32, 32, 3))
        x = Conv2D(channels[0], kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(1e-4))(inputs)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.relu)(x)

        for c in channels:
            for i in range(n):
                subsampling = i == 0 and c > 16
                strides = (2, 2) if subsampling else (1, 1)
                y = Conv2D(c, kernel_size=(3, 3), padding="same", strides=strides, kernel_initializer="he_normal",
                           kernel_regularizer=regularizers.l2(1e-4))(x)
                y = BatchNormalization()(y)
                y = Activation(tf.nn.relu)(y)
                y = Conv2D(c, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",
                           kernel_regularizer=regularizers.l2(1e-4))(y)
                y = BatchNormalization()(y)
                if subsampling:
                    x = Conv2D(c, kernel_size=(1, 1), strides=(2, 2), padding="same", kernel_initializer="he_normal",
                               kernel_regularizer=regularizers.l2(1e-4))(x)
                x = Add()([x, y])
                x = Activation(tf.nn.relu)(x)

        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        outputs = Dense(10, activation=tf.nn.softmax, kernel_initializer="he_normal")(x)

        model = Model(inputs=inputs, outputs=outputs)

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # initially lr=1e-3, decay=5e-4

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        return model


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x