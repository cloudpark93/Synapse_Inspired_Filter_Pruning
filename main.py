import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

from config import config
from train_evaluate_retrain import training_model_cifar10, training_model_imagenet, model_evaluate, model_prediction
from utils.dataset_loader import dataset, get_dataset
from utils.pruning_conv_fc_filters import pruning_model
from utils.util_func import model_history_save_and_plot, get_flops_, get_params_
from model.model_architectures import model_type

from tensorflow.keras import backend as K


def main(args):
    """ Load dataset """
    if args.dataset == 'cifar10':
        x_train, x_test, y_train, y_test = dataset('cifar10')
        ds_valid = [x_test, y_test]
    elif args.dataset == 'imagenet':
        if args.mode == 'train' or args.mode == 'pruned_and_retrain':
            config.DATA_AUGMENTATION = True
        else:
            config.DATA_AUGMENTATION = False
        ds_train = get_dataset(args.dataset_dir, 'train', args.batch_size)
        ds_valid = get_dataset(args.dataset_dir, 'validation', args.batch_size)


    """ Select Mode """
    if args.mode == 'train':
        model = model_type(args.arch)
        model, history = args.dataset == 'cifar10' \
                               and training_model_cifar10(model, args, x_train, x_test, y_train, y_test) \
                               or  training_model_imagenet(model, args, ds_train, ds_valid)

        # Get loss & acc
        results = args.dataset == 'cifar10' \
                  and model_evaluate(args, model, ds_valid) \
                  or model_evaluate(args, model, ds_valid)
        np.savetxt('{}/{}_{}_epoch{}%.csv'.format(
            args.mode, args.arch, args.dataset, args.epochs), results, delimiter=',')

        # Save model
        model.save('{}/{}_{}_epoch{}_acc{}.h5'.format(args.mode, args.arch, args.dataset, args.epochs, results[1]))

        # Save training history
        model_history_save_and_plot(args, history)

        del model
        K.clear_session()

    elif args.mode == 'pruning_each_layer':
        model = load_model(args.load_model)
        pruning_model(args, model, ds_valid)

        del model
        K.clear_session()

    elif args.mode == 'pruning_sensitivity':
        model = load_model(args.load_model)
        pruning_model(args, model, ds_valid)

        del model
        K.clear_session()

    elif args.mode == 'pruned_and_retrain':
        # pruned_model = load_model('pruning_sensitivity/{}_{}_pruned_with_sensitivity_{}_{}.h5'.format(args.arch, args.dataset, args.pruning_sensitivity, args.pruning_method))

        pruned_model = load_model(args.load_model)
        model, history = args.dataset == 'cifar10' \
                               and training_model_cifar10(pruned_model, args, x_train, x_test, y_train, y_test) \
                               or  training_model_imagenet(pruned_model, args, ds_train, ds_valid)

        # Save model
        pruned_model.save('{}/{}/{}_{}_epoch{}_sens{}.h5'.format(args.mode, args.pruning_method, args.arch, args.dataset, args.epochs, args.pruning_sensitivity))

        # Save training history
        model_history_save_and_plot(args, history)

        flops_ = get_flops_(pruned_model)
        params_ = get_params_(pruned_model)
        print("FLOPs : ", flops_)
        print("Params : ", params_)

        np.savetxt('{}/{}/{}_{}_epoch{}_param_{}.csv'.format(args.mode, args.pruning_method, args.arch, args.dataset, args.epochs, args.pruning_method), [flops_, params_],
                   delimiter=',')

        del pruned_model
        K.clear_session()
        
    elif args.mode == 'lr_finder':
        
        model = model_type(args.arch)
        model, history = args.dataset == 'cifar10' \
                               and training_model_cifar10(model, args, x_train, x_test, y_train, y_test) \
                               or  training_model_imagenet(model, args, ds_train, ds_valid)

        del model
        K.clear_session()   


if __name__ == '__main__':
    """ Main function """

    # Set arguments0
    parser = argparse.ArgumentParser(description='DYJS pruning on CIFAR10 & ImageNet datset using Keras')
    parser.add_argument('--mode', type=str, default='pruning_sensitivity', help='train, pruning_each_layer, pruning_sensitivity, pruned_and_retrain')
    parser.add_argument('--arch', type=str, default='vgg16', help='vgg16, resnet18, resnet56')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default 500)')
    parser.add_argument('--period', type=int, default=50, help='checkpoint (default 50)')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_min', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_max', type=float, default=0.076, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_drop', type=int, default=20, help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')

    parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset: cifar10, imagenet')
    parser.add_argument('--num_classes', type=str, default='10', help='number of class for training dataset (default: 10)')

    parser.add_argument('--pruning_method', type=str, default='DYJS_score', help='L1norm, GM, DYJS_score, DYJS_step, DYJS_step_gm')
    parser.add_argument('--pruning_percent', type=int, default=5, help='% of the filters are to be pruned (default: 5)')
    parser.add_argument('--pruning_sensitivity', type=int, default=70, help='% of the pruning sensitivity (default: 90)')

    parser.add_argument('--dataset_dir', type=str, default=config.DEFAULT_DATASET_DIR)
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--data_path', type=str, default='datas', help='Path of datasets (default: datas)')
    parser.add_argument('--load_model', type=str, default='vgg16_cifar10-450-0.93.h5', help='Path of model')
    parser.add_argument('--save_path', type=str, default='./pruned_and_retrain', help='Results save path (default: ./train_model_storage)')
    args = parser.parse_args()

    # GPU device connection
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    main(args)

    # # Start Main
    # method = ['DYJS_score', 'DYJS_step', 'DYJS_step_gm', 'L1norm']
    # mode = ['pruning_each_layer', 'pruning_sensitivity', 'pruned_and_retrain']
    #
    # for k in range(0, 3):
    #     args.mode = mode[k]
    #     for i in range(0, 4):
    #         args.pruning_method = method[i]
    #         if k == 0:
    #             main(args)
    #         else:
    #             for p in range(9, 4, -1):
    #                 args.pruning_sensitivity = p*10
    #                 print('Pruning method: ' + args.pruning_method)
    #                 main(args)


