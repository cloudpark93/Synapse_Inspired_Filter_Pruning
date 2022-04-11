import numpy as np
from scipy.spatial import distance


def distance_cal_func(weight, distance_method):
    if distance_method == "euclidean":
        similar_matrix = distance.cdist(weight, weight, 'euclidean')
    elif distance_method == "mahalanobis":
        similar_matrix = distance.cdist(weight, weight, 'mahalanobis')

    return similar_matrix


def geometric_median_DScore(weight, distance_method, prune_size):

    weight = weight.reshape(-1, weight.shape[-1])
    weight = np.transpose(weight)

    weight_pos = []
    weight_neg = []
    for i in range(weight.shape[0]):
        weight_pos.append(pos_DScore(weight[i]))
        weight_neg.append(neg_DScore(weight[i]))

    similar_matrix_pos = distance_cal_func(weight_pos, distance_method)
    similar_matrix_neg = distance_cal_func(weight_neg, distance_method)

    similar_sum_pos = (np.sum(similar_matrix_pos, axis=0))
    similar_sum_neg = (np.sum(similar_matrix_neg, axis=0))

    similar_sum_pos_dict = {}
    similar_sum_neg_dict = {}
    for i in range(weight.shape[0]):
        weight_number = 'weight_{}'.format(i)
        similar_sum_pos_dict[weight_number] = similar_sum_pos[i]
        similar_sum_neg_dict[weight_number] = similar_sum_neg[i]

    similar_sum_pos_dict_sort = sorted(similar_sum_pos_dict.items(), key=lambda kv: kv[1])
    similar_sum_neg_dict_sort = sorted(similar_sum_neg_dict.items(), key=lambda kv: kv[1])

    similar_sum_pos_dict_key = list(dict(similar_sum_pos_dict_sort).keys())
    similar_sum_neg_dict_key = list(dict(similar_sum_neg_dict_sort).keys())

    buff = []
    similar_array = []
    for i in range(prune_size, weight.shape[0]):
        similar_array = list(set(similar_sum_pos_dict_key[:i]) & set(similar_sum_neg_dict_key[:i]))
        if len(similar_array) >= prune_size:
            # scoring
            diff = len(similar_array) - prune_size + 1
            if diff == 1:
                break
            else:
                temp_arr = similar_array
                [temp_arr.remove(k) for k in buff] # temp_array = simultanously selected filter
                score = [
                    similar_sum_pos_dict_key.index(similar_array[k]) + similar_sum_neg_dict_key.index(similar_array[k])
                    for k in range(len(similar_array) - diff, len(similar_array))]
                temp_arr = np.array(temp_arr)
                temp_arr = temp_arr[np.array(score).argsort()]
                [similar_array.remove(temp_arr[k]) for k in range(diff - 1, 0, -1)]
                break
        else:
            buff = similar_array
            similar_array = []

    [similar_array.append(i) for i in buff]

    return similar_array


def pos(lst):
    return [x for x in lst if x >= 0] or None

def neg(lst):
    return [x for x in lst if x < 0] or None

def pos_DScore(lst):
    return [x if x > 0 else 0 for x in lst] or None

def neg_DScore(lst):
    return [x if x < 0 else 0 for x in lst] or None

