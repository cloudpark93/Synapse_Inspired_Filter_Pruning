import numpy as np
from numpy import linalg as LA
from kerassurgeon import Surgeon
from .geometric_method import geometric_median_DScore

def pos(lst):
    return [x for x in lst if x >= 0]

def neg(lst):
    return [x for x in lst if x < 0]


def pruning_method_conv(model, layer_to_prune, pruning_amount, method):

    if method == 'L1norm':
        # Load surgeon package
        surgeon = Surgeon(model)

        # Store weights from conv layers [0][1] = [weight][bias] Hence, [0] to store weights only
        conv_layer_weights = np.array([model.layers[i].get_weights()[0] for i in layer_to_prune])

        for i in range(len(conv_layer_weights)):
            if pruning_amount[i] == 0:
                continue

            weight = conv_layer_weights[i]

            sum_filters = LA.norm(weight.reshape(-1, weight.shape[3]), axis=0, ord=1)
            sort_sum_filters = np.sort(sum_filters)
            remove_channel = np.argwhere(sum_filters <= sort_sum_filters[pruning_amount[i] - 1]).flatten()
            print(remove_channel)

            # delete filters with lowest scores
            surgeon.add_job('delete_channels', model.layers[layer_to_prune[i]], channels=remove_channel)

        model_pruned = surgeon.operate()

        return model_pruned

    if method == 'D_score': 
        # Load surgeon package
        surgeon = Surgeon(model)

        # Store weights from conv layers [0][1] = [weight][bias] Hence, [0] to store weights only
        conv_layer_weights = [model.layers[i].get_weights()[0] for i in layer_to_prune]

        for i in range(len(conv_layer_weights)):
            if pruning_amount[i] == 0:
                continue
            weight = conv_layer_weights[i]
            index = np.array(range(weight.shape[-1]))

            pos_sum_filters = LA.norm(weight.reshape(-1, weight.shape[3]).clip(min=0), axis=0, ord=1)
            sort_pos_sum_filters = pos_sum_filters.argsort()

            neg_sum_filters = LA.norm(weight.reshape(-1, weight.shape[3]).clip(max=0), axis=0, ord=1)
            sort_neg_sum_filters = neg_sum_filters.argsort()

            pos_sum_filters[sort_pos_sum_filters[index]] = neg_sum_filters[sort_neg_sum_filters[index]] = index
            scored_sum_filters = pos_sum_filters + neg_sum_filters

            remove_channel = scored_sum_filters.argsort()[:pruning_amount[i]]
            print(remove_channel)

            # delete filters with lowest scores
            surgeon.add_job('delete_channels', model.layers[layer_to_prune[i]], channels=remove_channel)

        model_pruned = surgeon.operate()

        return model_pruned

    if method == 'D_step':
        # Load surgeon package
        surgeon = Surgeon(model)

        # Store weights from conv layers [0][1] = [weight][bias] Hence, [0] to store weights only
        conv_layer_weights = [model.layers[i].get_weights()[0] for i in layer_to_prune]

        for i in range(len(conv_layer_weights)):
            if pruning_amount[i] == 0:
                continue
            weight = conv_layer_weights[i]

            pos_sum_filters = LA.norm(weight.reshape(-1, weight.shape[3]).clip(min=0), axis=0, ord=1)
            sort_pos_sum_filters = pos_sum_filters.argsort()

            neg_sum_filters = LA.norm(weight.reshape(-1, weight.shape[3]).clip(max=0), axis=0, ord=1)
            sort_neg_sum_filters = neg_sum_filters.argsort()

            result = np.array([np.intersect1d(sort_pos_sum_filters[:j], sort_neg_sum_filters[:j]) for j in range(weight.shape[3])
                      if len(np.intersect1d(sort_pos_sum_filters[:j], sort_neg_sum_filters[:j])) >= pruning_amount[i]
                      and len(np.intersect1d(sort_pos_sum_filters[:j], sort_neg_sum_filters[:j])) <= pruning_amount[i] + 1])

            for k in range(len(result)):
                if len(result[k]) == pruning_amount[i]:
                    remove_channel = result[k]
                    break
                elif len(result[k]) == pruning_amount[i] + 1:
                    remove_channel = (result[k][(sort_pos_sum_filters[result[k]] + sort_neg_sum_filters[result[k]]).argsort()])[:-1]
                    break
            print(remove_channel)

            surgeon.add_job('delete_channels', model.layers[layer_to_prune[i]], channels=remove_channel)

        model_pruned = surgeon.operate()

        return model_pruned

    elif method == 'D_step_gm': #geometric_median_applied method
        # Load surgeon package
        surgeon = Surgeon(model)

        # Store weights from conv layers [0][1] = [weight][bias] Hence, [0] to store weights only
        conv_layer_weights = [model.layers[i].get_weights()[0] for i in layer_to_prune]

        for i in range(len(conv_layer_weights)):
            if pruning_amount[i] == 0:
                continue
            weight = conv_layer_weights[i]
            num_filters = len(weight[0, 0, 0, :])
            weight_removable = {}

            # 1. Reduce dimension 4D -> 2D
            # 2. Normalization 
            # 3. Sort norm value => get index order
            # 4. Sort weight by index order
            # 5. Calculate distance between coordinates
            # 6. Sum distance (of each filter)
            norm_val = geometric_median_DScore(weight, "euclidean", pruning_amount[i])
            print('distance calculation result: ', norm_val)
            remove_channel = [int(norm_val[i].split("_")[1]) for i in range(0, pruning_amount[i])]
            print(remove_channel)

            surgeon.add_job('delete_channels', model.layers[layer_to_prune[i]], channels=remove_channel)

        model_pruned = surgeon.operate()

        return model_pruned


