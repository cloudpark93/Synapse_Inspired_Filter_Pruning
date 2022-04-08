import numpy as np

# import keras
import os, sys, time, re, gc
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from utils.util_func import check_path
from keras.optimizers import SGD
from tensorflow.keras.applications import resnet, vgg16, vgg19
from tensorflow.keras.utils import to_categorical
from glob import glob
from pathlib import Path
from custom_lr_scheduler import CustomDecay
from lr_finder.keras_callback import LRFinder

def training_model_cifar10(model, args, x_train, x_test, y_train, y_test): 
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True)  # randomly flip images
        
    datagen.fit(x_train)    
        
    filename = args.arch + '{epoch:02d}-{val_accuracy:.4f}.h5'
    model_path = args.mode + '/' + filename
    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', verbose=1, save_weights_only=False, period=args.period)

    if args.mode == 'train':
        tensorboard = TensorBoard(log_dir=check_path('./tensorboard'))
        def lr_scheduler(epoch):
            lr = args.lr
            if epoch > 160:
                lr *= 0.5e-2
            elif epoch > 120:
                lr *= 1e-2
            elif epoch > 80:
                lr *= 1e-1
            return lr
        reduce_lr = LearningRateScheduler(lr_scheduler)
        # reduce_lr = LearningRateScheduler(lr_scheduler)
        
        
        model.compile(optimizer=tf.keras.optimizers.SGD(
            learning_rate=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
            
        history = model.fit_generator(datagen.flow(x=x_train, y=y_train, batch_size=args.batch_size),
                                  steps_per_epoch=x_train.shape[0]//args.batch_size,
                                  epochs=args.epochs,
                                  verbose=1,
                                  validation_data=(x_test, y_test))
            
    elif args.mode == 'lr_finder':
        lr_decayed_fn = LRFinder(min_lr=1e-4, max_lr=1)
        history = model.fit_generator(datagen.flow(x=x_train, y=y_train, batch_size=args.batch_size),
                                  steps_per_epoch=x_train.shape[0]//args.batch_size,
                                  epochs=args.epochs,
                                  verbose=1,
                                  validation_data=(x_test, y_test))
                                  callbacks=[lr_decayed_fn, cb_checkpoint]) 
    
    else:
        history = model.fit_generator(datagen.flow(x=x_train, y=y_train, batch_size=args.batch_size),
                                  steps_per_epoch=x_train.shape[0]//args.batch_size,
                                  epochs=args.epochs,
                                  verbose=1,
                                  validation_data=(x_test, y_test))
                         
    return model, history


def training_model_imagenet(model, args, ds_train, ds_valid):
    # tensorboard = TensorBoard(log_dir='./tensorboard')

    filename = args.arch + '{epoch:02d}-{val_accuracy:.4f}.h5'
    model_path = args.save_path + '/' + args.pruning_method + '/' + filename
    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', verbose=1, save_weights_only=False, period=args.period)
        
    if args.mode == 'lr_finder':
        lr_decayed_fn = LRFinder(min_lr=1e-4, max_lr=1)

        history = model.fit(
            x=ds_train,
            steps_per_epoch=1281167 // args.batch_size,
            validation_data=ds_valid,
            validation_steps=50000 // args.batch_size,
            callbacks=[lr_decayed_fn, cb_checkpoint], # tensorboard
            # The following doesn't seem to help in terms of speed.
            # use_multiprocessing=True, workers=4,
            epochs=args.epochs)
        
        
    elif args.mode == 'pruned_and_retrain':
        history = model.fit(
            x=ds_train,
            steps_per_epoch=1281167 // args.batch_size,
            validation_data=ds_valid,
            validation_steps=50000 // args.batch_size,
            # The following doesn't seem to help in terms of speed.
            # use_multiprocessing=True, workers=4,
            epochs=args.epochs)


    return model, history


def model_evaluate(args, model, ds_valid):
    # CIFAR-10 trained models
    # results = model.evaluate(ds_valid[0], ds_valid[1], batch_size=args.batch_size, verbose=1)

    # ImageNet trained models
    results = model.evaluate(x=ds_valid, steps=50000 // args.batch_size, verbose=1)
    return results


def top_k_accuracy(y_true, y_pred, k=1, tf_enabled=True):
    if tf_enabled:
        argsorted_y = tf.argsort(y_pred)[:,-k:]
        matches = tf.cast(tf.math.reduce_any(tf.transpose(argsorted_y) == tf.argmax(y_true, axis=1, output_type=tf.int32), axis=0), tf.float32)
        return tf.math.reduce_mean(matches).numpy()
    else:
        argsorted_y = np.argsort(y_pred)[:,-k:]
        return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()


def model_prediction(model_pruned):
    path_imagenet_val_dataset = Path("D:/ImageNet/data/")
    x_val_paths = glob(str(path_imagenet_val_dataset / "x_val*.npy"))
    
    # Sort filenames in ascending order
    x_val_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
    y_val = np.load(str(path_imagenet_val_dataset / "y_val.npy"))
    y_val_one_hot = to_categorical(y_val, 1000)

    y_pred = None
    for i, x_val_path in enumerate(x_val_paths):
        x_val = np.load(x_val_path).astype('float32')  # loaded as RGB
        y_pred_sharded = model_pruned.predict(x_val, verbose=0, use_multiprocessing=True, batch_size=64, callbacks=None)

        try:
            y_pred = np.concatenate([y_pred, y_pred_sharded])
        except ValueError:
            y_pred = y_pred_sharded

        del x_val
        gc.collect()

        completed_percentage = (i + 1) * 100 / len(x_val_paths)
        if completed_percentage % 5 == 0:
            print("{:5.1f}% completed.".format(completed_percentage))

    return top_k_accuracy(y_val_one_hot, y_pred, k=1), top_k_accuracy(y_val_one_hot, y_pred, k=5)