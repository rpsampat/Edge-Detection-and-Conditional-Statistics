import numpy as np


def matrix_build(x2, y2, uavg, vavg):

    x2uniq = np.unique(x2)
    y2uniq = np.unique(y2)
    S = np.zeros((len(y2uniq), len(x2uniq), 4))
    counts_x = np.histogram(x2, bins=len(x2uniq))[0]
    avg = np.mean(counts_x)
    counts_y = np.histogram(y2, bins=len(y2uniq))[0]
    avg_y = np.mean(counts_y)
    y_frequency_from_x = int(np.floor(avg))
    x_frequency_from_y = int(np.floor(avg_y))
    index_of_yuniq = np.where(counts_y >= x_frequency_from_y)[0]
    index_of_xuniq = np.where(counts_x >= y_frequency_from_x)[0]
    x_list = np.zeros(len(x2))
    y_list = np.zeros(len(y2))
    index1 = 0
    index2 = 0

    for iter in range(len(x2uniq)):
        if iter in index_of_xuniq:
            x_list[index1] = x2uniq[iter]
            index1 += 1

    for iter2 in range(len(y2uniq)):
        if iter2 in index_of_yuniq:
            y_list[index2] = y2uniq[iter2]
            index2 += 1

    for iter in range(len(x2)):
        S_index_x = np.where(x2uniq == x2[iter])[0]
        S_index_y = np.where(y2uniq == y2[iter])[0]
        S[S_index_y, S_index_x, 0] = x2[iter]
        S[S_index_y, S_index_x, 1] = y2[iter]
        S[S_index_y, S_index_x, 2] = uavg[iter]
        S[S_index_y, S_index_x, 3] = vavg[iter]

    x_list = x_list[:index1]
    y_list = y_list[:index2]
    min_X = np.where(x2uniq == min(x_list))[0][0]
    max_X = np.where(x2uniq == max(x_list))[0][0]
    min_Y = np.where(y2uniq == min(y_list))[0][0]
    max_Y = np.where(y2uniq == max(y_list))[0][0]
    S = S[min_Y:max_Y + 1, min_X:max_X + 1, :]

    return S, x_list, y_list, index_of_xuniq, index_of_yuniq
