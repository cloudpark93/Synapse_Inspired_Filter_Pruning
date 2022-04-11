import numpy as np
from kerassurgeon import Surgeon
from .geometric_method import geometric_median, geometric_median_DYJS

def pos(lst):
    return [x for x in lst if x >= 0] or None

def neg(lst):
    return [x for x in lst if x < 0] or None


def pruning_method_fc(model, layer_to_prune, pruning_amount, method):

    if method == 'L1norm':
        # Load surgeon package
        surgeon = Surgeon(model)

        # Store weights from conv layers [0][1] = [weight][bias] Hence, [0] to store weights only
        fc_layer_weights = [model.layers[i].get_weights()[0] for i in layer_to_prune]

        for i in range(len(fc_layer_weights)):
            if pruning_amount[i] == 0:
                continue
            weight = fc_layer_weights[i]
            num_filters = weight.shape[1]
            weight_removable = {}

            # compute L1-nom of each filter weight and store it in a dictionary(weight_removable)
            for j in range(num_filters):
                L1_norm = np.sum(abs(weight[:, j]))
                filter_number = 'filter_{}'.format(j)
                weight_removable[filter_number] = L1_norm

            # sort the filter according to the ascending L1 value
            weight_removable_sort = sorted(weight_removable.items(), key=lambda kv: kv[1])

            # extracting filter number from '(filter_2, 0.515..), eg) extracting '2' from '(filter_2, 0.515..)
            remove_channel = [int(weight_removable_sort[i][0].split("_")[1]) for i in range(0, pruning_amount[i])]
            print(remove_channel)

            # delete filters with lowest scores
            surgeon.add_job('delete_channels', model.layers[layer_to_prune[i]], channels=remove_channel)

        model_pruned = surgeon.operate()

        return model_pruned

    if method == 'D_score': #filter_ranking_score
        # Load surgeon package
        surgeon = Surgeon(model)

        # Store weights from conv layers [0][1] = [weight][bias] Hence, [0] to store weights only
        fc_layer_weights = [model.layers[i].get_weights()[0] for i in layer_to_prune]

        for i in range(len(fc_layer_weights)):
            if pruning_amount[i] == 0:
                continue
            weight = fc_layer_weights[i]
            num_filters = weight.shape[1]

            filter_sum_positive = {}
            filter_sum_negative = {}
            filter_sum_positive_score = {}
            filter_sum_negative_score = {}
            filter_sum_ranking_score = {}

            ##########
            # Filter sum positive & negative magnitude
            ##########
            for j in range(num_filters):
                flatten_filter = weight[:, j].flatten()

                filter_number = 'filter_{}'.format(j)
                filter_sum_positive[filter_number] = np.sum(pos(flatten_filter))
                filter_sum_negative[filter_number] = abs(np.sum(neg(flatten_filter)))

            ##########
            # Sorting positive & negative sum
            ##########
            # key = operator doesn't work. so used key = lambda, kv - key value, kv[0]: sort by name, kv[1]: sort by value
            filter_sum_positive_sort = sorted(filter_sum_positive.items(), key=lambda kv: kv[1])
            filter_sum_negative_sort = sorted(filter_sum_negative.items(), key=lambda kv: kv[1]) 
            filter_positive_key = list(dict(filter_sum_positive_sort).keys())
            filter_negative_key = list(dict(filter_sum_negative_sort).keys())

            ##########
            # Score positive & negative sum
            ##########
            for k in range(num_filters):
                filter_sum_positive_score[filter_positive_key[k]] = k
                filter_sum_negative_score[filter_negative_key[k]] = k

            ##########
            # Ranking positive & negative sum
            ##########
            for l in range(num_filters):
                filter_number = 'filter_{}'.format(l)
                filter_sum_ranking_score[filter_number] = filter_sum_positive_score[filter_number] + \
                                                          filter_sum_negative_score[filter_number]
            filter_sum_ranking_score_sort = sorted(filter_sum_ranking_score.items(), key=lambda kv: kv[1])

            # extracting filter number from '(filter_2, 0.515..), eg) extracting '2' from '(filter_2, 0.515..)
            remove_channel = [int(filter_sum_ranking_score_sort[i][0].split("_")[1]) for i in
                              range(0, pruning_amount[i])]
            print(remove_channel)

            # delete filters with lowest scores
            surgeon.add_job('delete_channels', model.layers[layer_to_prune[i]], channels=remove_channel)

        model_pruned = surgeon.operate()

        return model_pruned

    if method == 'D_step': 
        # Load surgeon package
        surgeon = Surgeon(model)

        # Store weights from conv layers [0][1] = [weight][bias] Hence, [0] to store weights only
        fc_layer_weights = [model.layers[i].get_weights()[0] for i in layer_to_prune]

        for i in range(len(fc_layer_weights)):
            if pruning_amount[i] == 0:
                continue
            weight = fc_layer_weights[i]
            num_filters = weight.shape[1]

            filter_sum_positive = {}
            filter_sum_negative = {}

            ##########
            # Filter sum positive & negative magnitude
            ##########
            for j in range(num_filters):
                flatten_filter = weight[:, j].flatten()

                filter_number = 'filter_{}'.format(j)
                filter_sum_positive[filter_number] = np.sum(pos(flatten_filter))
                filter_sum_negative[filter_number] = abs(np.sum(neg(flatten_filter)))

            ##########
            # Sorting positive & negative sum
            ##########
            # key = operator doesn't work. so used key = lambda, kv - key value, kv[0]: sort by name, kv[1]: sort by value
            filter_sum_positive_sort = sorted(filter_sum_positive.items(), key=lambda kv: kv[1])
            filter_sum_negative_sort = sorted(filter_sum_negative.items(), key=lambda kv: kv[1]) 

            filter_positive_key = list(dict(filter_sum_positive_sort).keys())
            filter_negative_key = list(dict(filter_sum_negative_sort).keys())

            buff = []
            similar_array = []
           
            for j in range(pruning_amount[i], num_filters):
                similar_array = list(set(filter_positive_key[:j]) & set(filter_negative_key[:j]))

                if len(similar_array) >= pruning_amount[i]:
                    # scoring
                    diff = len(similar_array) - pruning_amount[i] + 1
                    if diff == 1:
                        break
                    else:
                        temp_arr = similar_array
                        [temp_arr.remove(k) for k in buff]
                        score = [
                            filter_positive_key.index(similar_array[k]) + filter_negative_key.index(similar_array[k])
                            for k in range(len(similar_array) - diff, len(similar_array))]
                        temp_arr = np.array(temp_arr)
                        temp_arr = temp_arr[np.array(score).argsort()]
                        [similar_array.remove(temp_arr[k]) for k in range(diff - 1, 0, -1)]
                        break
                else:
                    buff = similar_array
                    similar_array = []

            [similar_array.append(i) for i in buff]
            print('similar array to remove is ', similar_array)

            remove_channel = [int(similar_array[i].split("_")[1]) for i in range(0, pruning_amount[i])]
            print(remove_channel)

            # delete filters with lowest scores
            surgeon.add_job('delete_channels', model.layers[layer_to_prune[i]], channels=remove_channel)

        model_pruned = surgeon.operate()

        return model_pruned

    elif method == 'D_step_gm': # geometric_median_conv
        # Load surgeon package
        surgeon = Surgeon(model)

        # Store weights from conv layers [0][1] = [weight][bias] Hence, [0] to store weights only
        fc_layer_weights = [model.layers[i].get_weights()[0] for i in layer_to_prune]

        for i in range(len(fc_layer_weights)):
            if pruning_amount[i] == 0:
                continue
            weight = fc_layer_weights[i]
            num_filters = weight.shape[1]
            weight_removable = {}

            # 1. Reduce dimension 4D -> 2D
            # 2. Normalization (L1, L2, BYJS)
            # 3. Sort norm value => get index order
            # 4. Sort weight by index order
            # 5. Calculate distance between coordinates
            # 6. Sum distance (of each filter)
            # norm_val = geometric_median(weight, "euclidean", "DYJS")
            norm_val = geometric_median_DYJS(weight, "euclidean", pruning_amount[i])
            print('distance calculation result: ', norm_val)
            remove_channel = [int(norm_val[i].split("_")[1]) for i in range(0, pruning_amount[i])]
            print(remove_channel)

            surgeon.add_job('delete_channels', model.layers[layer_to_prune[i]], channels=remove_channel)

        model_pruned = surgeon.operate()

        return model_pruned