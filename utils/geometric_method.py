
import numpy as np
from scipy.spatial import distance

def L1norm_method(weight):
    return np.linalg.norm(weight, 1, axis=1)

def L2norm_method(weight):
    return np.linalg.norm(weight, 2, axis=1)

def DYJSnorm_method(weight):
    filter_sum_positive = {}
    filter_sum_negative = {}
    filter_sum_positive_score = {}
    filter_sum_negative_score = {}
    filter_sum_ranking_score = []

    for j in range(weight.shape[0]):
        filter_number = 'filter_{}'.format(j)
        filter_sum_positive[filter_number] = np.sum(pos(weight[j]))
        filter_sum_negative[filter_number] = abs(np.sum(neg(weight[j])))

    filter_sum_positive_sort = sorted(filter_sum_positive.items(), key=lambda kv: kv[1])
    filter_sum_negative_sort = sorted(filter_sum_negative.items(), key=lambda kv: kv[1])
    filter_positive_key = list(dict(filter_sum_positive_sort).keys())
    filter_negative_key = list(dict(filter_sum_negative_sort).keys())

    for k in range(weight.shape[0]):
        filter_sum_positive_score[filter_positive_key[k]] = k
        filter_sum_negative_score[filter_negative_key[k]] = k

    for l in range(weight.shape[0]):
        filter_number = 'filter_{}'.format(l)
        filter_sum_ranking_score.append(filter_sum_positive_score[filter_number] + filter_sum_negative_score[filter_number])

    return np.array(filter_sum_ranking_score)


def distance_cal_func(weight, distance_method):
    if distance_method == "euclidean":
        similar_matrix = distance.cdist(weight, weight, 'euclidean')
    elif distance_method == "mahalanobis":
        similar_matrix = distance.cdist(weight, weight, 'mahalanobis')

    return similar_matrix

# [Geometric Median with L1-norm process]
# > get model.parameters() => param.data() that is weight(3x3x3x64)
# > weight_vec = weight_torch.view(weight_torch.size()[0], -1)
# > norm = torch.norm(weight_vec, p=1, d=1)
# > filter_large_index = norm_np.argsort()[filter_pruned_num:]
# > indices = torch.LongTensor(filter_large_index).cuda()
# > weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
# > weight 형태를변환(distance 함수에 대입하기 위해서) > distance.cdist - distance를 구하는 함수
def geometric_median(weight, distance_method, norm_method):
    # To calculate distance between coordinates weights shape has to change

    # Example is 3D to 2D but its actually 4D to 2D
    # weight = array([[[ 7.,  6.,  7.],
    #                  [24., 27., 30.],
    #                  [51., 54., 57.]],
    #
    #                 [[24., 27., 30.],
    #                  [ 7.,  6.,  7.],
    #                  [51., 54., 57.]],
    #
    #                 [[ 7.,  6.,  7.],
    #                  [24., 27., 30.],
    #                  [51., 54., 57.]]], dtype=float32)

    ###########
    # Reduce dimension 4D -> 2D (to calculate distance)
    ###########
    # weight = array([[ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.],
    #                 [-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.],
    #                 [14., 15., 16., 17., 18., 19., 20., 21., 22.]], dtype=float32)
    # Weight : 3x3x3x64 => 27 x 64 => 64 x 27 (to rank each filter)
    weight = weight.reshape(-1, weight.shape[3])
    weight = np.transpose(weight)


    ###########
    # Normalization
    ###########
    # np.linalg.norm(input matrix,
    #                Order of the norm (see the link),
    #                specifies the axis of x along which to compute the vector norms)
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    weight_norm = _methods[norm_method](weight)
    # L1 => array([81., 20., 162.], dtype=float32)
    # L2 => array([28.089144,  7.745967, 54.552727], dtype=float32)


    ###########
    # Sort norm value => get index order
    ###########
    # get sorted filter index (int64)
    # [81., 20., 162.].argsort() => get index array [1, 0, 2]
    weight_sorted_index = weight_norm.argsort()


    ###########
    # Sort weight by index order
    ###########
    #                                                                                [order]
    # weight = array([[-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.],                    0
    #                 [ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.],                    1
    #                 [14., 15., 16., 17., 18., 19., 20., 21., 22.]], dtype=float32)    2
    # change matrix arrays in index order (torch.index_select)
    weight = weight[weight_sorted_index]


    ###########
    # Calculate distance between coordinates
    ###########
    # for euclidean/mahalanobis distance
    similar_matrix = distance_cal_func(weight, distance_method)


    ###########
    # Sum distance
    ###########
    similar_sum = np.sum(np.abs(similar_matrix), axis=0)

    return similar_sum

def geometric_median_DYJS(weight, distance_method, prune_size):

    weight = weight.reshape(-1, weight.shape[-1])
    weight = np.transpose(weight)

    weight_pos = []
    weight_neg = []
    for i in range(weight.shape[0]):
        weight_pos.append(pos_DYJS(weight[i]))
        weight_neg.append(neg_DYJS(weight[i]))

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
                [temp_arr.remove(k) for k in buff] # 이 부분에서 temp_array = 동시에 걸리는 두가지 filter
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

def pos_DYJS(lst):
    return [x if x > 0 else 0 for x in lst] or None

def neg_DYJS(lst):
    return [x if x < 0 else 0 for x in lst] or None

_methods = {
    'L1': L1norm_method,
    'L2': L2norm_method,
    'DYJS': DYJSnorm_method
}