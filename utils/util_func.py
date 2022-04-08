import os
import pandas as pd
import matplotlib.pyplot as plt

from keras_flops import get_flops
from keras.layers import Conv2D, Dense


########################################################################################################################
#                                       Function: Calculate FLOPs & Params                                             #
########################################################################################################################

def get_flops_(model):
    return get_flops(model, batch_size=1)

def get_params_(model):
    return model.count_params()


########################################################################################################################
#                                     Function: Checking Conv & FC Layers                                              #
########################################################################################################################

def check_conv2d_layers(model):
    count = -1
    layer_to_prune_original_model_conv = []
    layer_to_prune_for_continuous_pruning_conv = []
    original_num_filters_conv = []

    for layer in model.layers:
        count = count + 1
        print(layer.name)
        if count == 3:
            layer_to_prune_original_model_conv.append(count)
            original_num_filters_conv.append(layer.weights[0].shape[3])
        elif isinstance(layer, Conv2D) and layer.name.split('_')[2] == 'conv1':
            layer_to_prune_original_model_conv.append(count)
            original_num_filters_conv.append(layer.weights[0].shape[3])

    return layer_to_prune_original_model_conv, layer_to_prune_for_continuous_pruning_conv, original_num_filters_conv


def check_fc_layers(model):
    count = -1
    layer_to_prune_original_model_fc = []
    layer_to_prune_for_continuous_pruning_fc = []
    original_num_filters_fc = []

    for layer in model.layers:
        count += 1
        if isinstance(layer, Dense):
            layer_to_prune_original_model_fc.append(count)
            layer_to_prune_for_continuous_pruning_fc.append(count+1)
            original_num_filters_fc.append(layer.weights[0].shape[1])

    # del layer_to_prune_original_model_fc[-1]
    # del layer_to_prune_for_continuous_pruning_fc[-1]
    # del original_num_filters_fc[-1]


    return layer_to_prune_original_model_fc, layer_to_prune_for_continuous_pruning_fc, original_num_filters_fc


########################################################################################################################
#                                               Function: Plot graphs                                                  #
########################################################################################################################

def model_history_save_and_plot(args, history):
    # convert the history.history dictionary to a pandas DataFrame and save it as csv
    history_df = pd.DataFrame(history.history)
    history_df_csv = args.save_path + 'history.csv'
    with open(history_df_csv, mode='w') as f:
        history_df.to_csv(f)

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{}/{}_{}_{}_{}_acc.jpg'.format(args.save_path, args.arch, args.dataset, args.mode, args.pruning_method), dpi=300)
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{}/{}_{}_{}_{}_loss.jpg'.format(args.save_path, args.arch, args.dataset, args.mode, args.pruning_method), dpi=300)
    plt.close()



########################################################################################################################
#                                               Function: Additional                                                   #
########################################################################################################################

def check_path(path):
    temp_file = '.'
    for file in path.split('/'):
        temp_file = temp_file + '/' + file
        if not os.path.isdir(temp_file):
            os.mkdir(temp_file)

    return path
