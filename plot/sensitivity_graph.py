import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib import colors as mcolors
from keras.models import load_model
from pathlib import Path
from glob import glob
from tensorflow.keras.utils import to_categorical
import os, sys, time, re, gc
from keras.layers import Conv2D

count = -1
layer_to_prune_original_model_conv = []
layer_to_prune_for_continuous_pruning_conv = []
original_num_filters_conv = []
img_w, img_h = 224, 224
classes = 1000

# method = 'DYJS_score'
# method = 'DYJS_step'
method = 'DYJS_step_gm'
arch = 'resnet18'
dataset_ = 'imagenet'
dataset_path = 'D:/imagenet/ImageNet'
pruning_index_per = 0.05
predict_batch_size = 64


##### Setting Model #####
model = load_model('resnet18_imagenet_1000.h5')

##### Check Conv2D Layers #####
for layer in model.layers:
    count += 1
    if isinstance(layer, Conv2D):
        layer_to_prune_original_model_conv.append(count)
        layer_to_prune_for_continuous_pruning_conv.append(count + 1)
        original_num_filters_conv.append(layer.weights[0].shape[3])

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

pruning_index = 5.0
plot_line_color = ["r", "g", "b", "k", "y", "m", "c"]
plot_line_style = ["-", "--"]

root_file = ["DYJS_score", "DYJS_step", "DYJS_step_gm"]
file_sub_name = ["DYJS_score", "DYJS_step", "DYJS_step_gm"]
pruning_method = ["Dynamic Score base", "Dynamic Step base", "Dynamic Step GM base"]
epoch_arr = ["%", "%", "%"]

num_filters = original_num_filters_conv

layer_number = [3, 11, 21, 30, 40, 49, 59, 68, 78]

for method_count in range(3):
    top1_acc = []
    top5_acc = []
    file_number = sum([len(d) for r, d, files in os.walk("test_continuous_pruning/DYJS_score")])

    for layer in layer_number:
        pruned_top1_acc = []
        pruned_top5_acc = []

        for i in range(1, int(1/(pruning_index/100))):
            data = csv.DictReader(open(
                "test_continuous_pruning/{}/test_continuous_pruning_layer{}/{}_imagenet_after_prune_{}_{}{}.csv"
                .format(root_file[method_count], layer, arch, file_sub_name[method_count], pruning_index * i,
                        epoch_arr[method_count])))
            for raw in data:
                pruned_top1_acc.append(list(raw.keys()))
                pruned_top5_acc.append(list(raw.values()))

        if layer == 1:
            conv_layers_top1_acc = np.array(pruned_top1_acc).astype(np.float32)
        else:
            conv_layers_top1_acc = np.append(conv_layers_top1_acc, np.array(pruned_top1_acc).astype(np.float32), axis=1)


    ############################## ACC ##############################
    plt.style.use("ggplot")
    plt.figure("Acc figure {} method Conv layer{}".format(file_sub_name[method_count], layer))
    x = range(int(pruning_index), 100, int(pruning_index))
    for layer in range(0, file_number):
        plt.plot(x, conv_layers_top1_acc[:, layer], linestyle=plot_line_style[layer // len(plot_line_color)],
                 marker='o', color=plot_line_color[layer % len(plot_line_color)], label="conv_{} {}".format(layer+1, num_filters[layer]),
                 linewidth=1.0, markersize=2)

    plt.ylim(0.0, 0.8)
    plt.title("Imagenet {} Top-1 Accuracy \n{} method".format(arch, pruning_method[method_count]))
    plt.xlabel("Filters Pruned Away (%)")
    plt.ylabel("Accuracy")
    plt.legend(loc=3)
    plt.savefig('test_continuous_pruning/Imagenet {} Top-1 Accuracy {} method.jpg'.format(arch, pruning_method[method_count]), dpi=300)
    print('Conv layer accuracy image saved successfully')
    plt.close()

