import numpy as np

from keras import backend as K
from keras.optimizers import SGD

from .pruning_amount_extraction_based_on_sensitivity import conv_pruning_amount_calculator, fc_pruning_amount_calculator
from .util_func import check_conv2d_layers, check_fc_layers, check_path
from .pruning_method_conv import pruning_method_conv
from .pruning_method_fc import pruning_method_fc
from train_evaluate_retrain import model_evaluate, model_prediction

########################################################################################################################
#                                     Function: Conv & FC filter pruning                                               #
########################################################################################################################

def pruning_filters_conv(args, pruning_index, layer_to_prune, model_for_pruning, original_num_filters, method):

    pruning_amount = [int(original_num_filters[i] * pruning_index[i]) for i in range(len(original_num_filters))]

    model_pruned = pruning_method_conv(model_for_pruning, layer_to_prune, pruning_amount, method)

    sgd = SGD(lr=args.lr, decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    model_pruned.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model_pruned


def pruning_filters_fc(args, pruning_index, layer_to_prune, model_for_pruning, original_num_filters, method):

    pruning_amount = [int(original_num_filters[i] * pruning_index[i]) for i in range(len(original_num_filters))]

    model_pruned = pruning_method_fc(model_for_pruning, layer_to_prune, pruning_amount, method)

    sgd = SGD(lr=args.lr, decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    model_pruned.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model_pruned


def pruning_model(args, model, ds_valid):
    # Check Conv & FC layers to prune
    layer_to_prune_original_model_conv, layer_to_prune_for_continuous_pruning_conv, original_num_filters_conv = check_conv2d_layers(model)
    # layer_to_prune_original_model_fc, layer_to_prune_for_continuous_pruning_fc, original_num_filters_fc = check_fc_layers(model)
    pruning_index_fc = []
    layer_to_prune_for_continuous_pruning_fc = []

    # Set pruning percent
    pruning_index_per = 0.01 * args.pruning_percent  # 0.05 = 5% of the filters are to be pruned

    if args.mode == 'pruning_each_layer':
        # Pruning Conv layers
        pruning_index_conv = np.ones((len(layer_to_prune_original_model_conv),)) * pruning_index_per
        pruning_each_layer('conv', model, args, ds_valid, pruning_index_per, pruning_index_conv,
                    layer_to_prune_original_model_conv, layer_to_prune_for_continuous_pruning_conv, original_num_filters_conv)

        # Pruning FC layers
        # pruning_index_fc = np.ones((len(layer_to_prune_original_model_fc),)) * pruning_index_per
        # pruning_each_layer('fc', model, args, ds_valid, pruning_index_per, pruning_index_fc,
        #             layer_to_prune_original_model_fc, layer_to_prune_for_continuous_pruning_fc, original_num_filters_fc)

    elif args.mode == 'pruning_sensitivity':
        pruning_index_conv = conv_pruning_amount_calculator(args.pruning_method, float(args.pruning_percent), args.pruning_sensitivity/100.0)
        # pruning_index_fc = fc_pruning_amount_calculator(args.pruning_method, float(args.pruning_percent), args.pruning_sensitivity/100.0)

        pruning_model_with_sensitivity(args, model, ds_valid, pruning_index_conv, pruning_index_fc,
                                       layer_to_prune_original_model_conv, layer_to_prune_for_continuous_pruning_fc, original_num_filters_conv)



def pruning_each_layer(layer_type, model, args, ds_valid, pruning_index_per, pruning_index_temp, layer_to_prune_original_model, layer_to_prune_for_continuous_pruning, original_num_filters):
    # For pruning job
    for layer_to_prune in range(0, len(layer_to_prune_original_model)):
        check_path('test_continuous_pruning/{}/{}/test_continuous_pruning_layer{}'.format(args.pruning_method, layer_type, layer_to_prune+1))

        pruning_index = [pruning_index_temp[layer] if layer == layer_to_prune else 0 for layer in range(len(pruning_index_temp))]

        for i in range(1, int(1 / pruning_index_per)):
            if i == 1:
                # load model to prune
                model_pruned = model
                # prune & save first layer
                model_pruned = layer_type == 'conv' \
                               and pruning_filters_conv(args, pruning_index, layer_to_prune_original_model, model_pruned, original_num_filters, args.pruning_method) \
                               or pruning_filters_fc(args, pruning_index, layer_to_prune_original_model, model_pruned, original_num_filters, args.pruning_method)

            else:
                # prune & save layer
                model_pruned = layer_type == 'conv' \
                               and pruning_filters_conv(args, pruning_index, layer_to_prune_original_model, model_pruned, original_num_filters, args.pruning_method) \
                               or pruning_filters_fc(args, pruning_index, layer_to_prune_original_model, model_pruned, original_num_filters, args.pruning_method)

            # Evaluation after pruning
            results = model_evaluate(args, model, ds_valid)
            # results = args.dataset == 'cifar10' \
            #           and model_evaluate(args, model, ds_valid) \
            #           or model_prediction(model)
            np.savetxt('test_continuous_pruning/{}/{}/test_continuous_pruning_layer{}/{}_{}_after_prune_{}_{}%.csv'.format(
                    args.pruning_method, layer_type, layer_to_prune+1,
                    args.arch, args.dataset,
                    args.pruning_method, pruning_index_per * 100 * i), results, delimiter=',')

            K.clear_session()

        del model_pruned
        K.clear_session()


def pruning_model_with_sensitivity(args, model, ds_valid, pruning_index_conv, pruning_index_fc,
                                   layer_to_prune_original_model_conv, layer_to_prune_for_continuous_pruning_fc, original_num_filters):
    # conv layer pruning
    model_pruned = pruning_filters_conv(args, pruning_index_conv, layer_to_prune_original_model_conv, model, original_num_filters,
                                        args.pruning_method)

    # fc layer pruning
    # model_pruned = pruning_filters_fc(pruning_index_fc, layer_to_prune_for_continuous_pruning_fc, model_pruned,
    #                                 args.pruning_method)

    # results = model_pruned.evaluate(x_test, y_test, verbose=0)
    # np.savetxt('{}/{}_{}_pruned_with_sensitivity_{}.csv'.format(args.mode, args.arch, args.dataset, args.pruning_method), results, delimiter=',')
    # print('Test loss for pruned model: ', results[0])
    # print('Test accuracy for pruned model: ', results[1])
    model_pruned.save('{}/{}_{}_pruned_with_sensitivity_{}_{}.h5'.format(args.mode, args.arch, args.dataset, args.pruning_sensitivity, args.pruning_method))

    del model_pruned
    K.clear_session()