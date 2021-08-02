# UTILITY-FUNCTIONS (READING DATA)

from __future__ import print_function

import numpy as np
from src.utils import *

__all__ = ["read_train_data_f", "read_train_data_i", "read_train_data_r"]


def read_train_data_f(filename2read, dim_p, h_prev, h_post, idx_f_use, data_size, sp_x=False, sp_y=False, is_npz=False):
    """ Reads (feature) train-data (single). """
    idx_f_use = make_numpy_array(idx_f_use, keep_1dim=True)
    len_filename = len(filename2read)
    _data_size = 0
    _data_size_list = []
    for nidx_d in range(0, len_filename):
        filename2read_sel = filename2read[nidx_d]
        data_read_tmp = np.load(filename2read_sel, allow_pickle=True)
        f_train_in = data_read_tmp['data_f'] if is_npz else data_read_tmp[()]['data_f']
        _data_size = _data_size + f_train_in.shape[0]
        _data_size_list.append(f_train_in.shape[0])

    _ratio = data_size / float(_data_size)
    data_size_list = [int(x * _ratio) for x in _data_size_list]
    data_size_list[-1] = data_size - np.sum(data_size_list[0:-1])

    dim_x, dim_y = dim_p * h_prev, dim_p * h_post
    dim_x_3 = 3 * h_prev
    if sp_y:
        dim_y_3 = 3 * (h_post + 1)
    else:
        dim_y_3 = 3 * (h_post + 2)
    dim_f = idx_f_use.shape[0] if len(idx_f_use) > 0 else 0

    idx_xin_tmp, idx_yin_tmp = np.arange(0, dim_x_3), np.arange(0, dim_y_3)
    if sp_y:
        idx_y0 = np.arange(0, dim_p * h_post)
        idx_y1 = np.arange(dim_p, dim_p * (h_post + 1))
    else:
        idx_y0 = np.arange(dim_p, dim_p * (h_post + 1))
        idx_y1 = np.arange(dim_p * 2, dim_p * (h_post + 2))

    h_prev_ref, h_post_ref = dim_x_3, dim_y_3

    x_train = np.zeros((data_size, dim_x), dtype=np.float32)
    y0_train = np.zeros((data_size, dim_y), dtype=np.float32)
    y1_train = np.zeros((data_size, dim_y), dtype=np.float32)
    f_train = np.zeros((data_size, dim_f), dtype=np.float32) if dim_f > 0 else []

    cnt_data = 0
    idx_sel_list = []
    for nidx_d in range(0, len_filename):
        filename2read_sel = filename2read[nidx_d]
        data_read_tmp = np.load(filename2read_sel, allow_pickle=True)

        if is_npz:
            x_train_in = data_read_tmp['data_x_sp'] if sp_x else data_read_tmp['data_x']
            y_train_in = data_read_tmp['data_y_sp'] if sp_y else data_read_tmp['data_y']
        else:
            x_train_in = data_read_tmp[()]['data_x_sp'] if sp_x else data_read_tmp[()]['data_x']
            y_train_in = data_read_tmp[()]['data_y_sp'] if sp_y else data_read_tmp[()]['data_y']

        if nidx_d == 0:
            h_prev_ref, h_post_ref = x_train_in.shape[1], y_train_in.shape[1]

        if dim_p == 2:
            idx_xin_tmp = np.setdiff1d(idx_xin_tmp, np.arange(2, h_prev_ref, 3))
            idx_yin_tmp = np.setdiff1d(idx_yin_tmp, np.arange(2, h_post_ref, 3))

        x_train_in = x_train_in[:, idx_xin_tmp]
        y_train_in = y_train_in[:, idx_yin_tmp]

        if dim_f > 0:
            f_train_in = data_read_tmp['data_f'] if is_npz else data_read_tmp[()]['data_f']
            f_train_in = f_train_in[:, idx_f_use]

        # Update
        size_before = int(x_train_in.shape[0])
        idx_rand_tmp_ = np.random.permutation(size_before)
        size_after = int(data_size_list[nidx_d])
        idx_rand_tmp = idx_rand_tmp_[np.arange(0, size_after)]
        idx_sel_list.append(idx_rand_tmp)

        idx_update_tmp = np.arange(cnt_data, cnt_data + size_after)
        x_train[idx_update_tmp, :] = x_train_in[idx_rand_tmp, :]
        y_train_in_tmp = y_train_in[idx_rand_tmp, :]

        y0_train[idx_update_tmp, :] = y_train_in_tmp[:, idx_y0]
        y1_train[idx_update_tmp, :] = y_train_in_tmp[:, idx_y1]

        if dim_f > 0:
            f_train[idx_update_tmp, :] = f_train_in[idx_rand_tmp, :]

        cnt_data = cnt_data + size_after

    idx_update = np.arange(0, cnt_data)
    x_train = x_train[idx_update, :]
    y0_train = y0_train[idx_update, :]
    y1_train = y1_train[idx_update, :]

    if dim_f > 0:
        f_train = f_train[idx_update, :]
    else:
        f_train = []

    return x_train, y0_train, y1_train, f_train, idx_sel_list


def read_train_data_i(filename2read, idx_i_use, idx_sel_list, is_npz=False):
    """ Reads (image) train-data. """
    data_size = 0
    for nidx_d in range(0, len(idx_sel_list)):
        data_size = data_size + len(idx_sel_list[nidx_d])

    # Read 1st-data
    print(".", end='')
    filename2read_sel = filename2read[0]
    data_read_tmp = np.load(filename2read_sel, allow_pickle=True)
    i_train_in = data_read_tmp['data_i'] if is_npz else data_read_tmp[()]['data_i']
    i_train_in = i_train_in[:, :, :, idx_i_use]
    size_in = np.shape(i_train_in)
    size_in = size_in[1:]

    # Set i_train
    i_train = np.zeros((data_size, size_in[0], size_in[1], len(idx_i_use)), dtype=np.float16)
    cnt_data = 0

    len_sel = len(idx_sel_list[0])
    idx_update_tmp = np.arange(cnt_data, cnt_data + len_sel)
    i_train[idx_update_tmp, :, :, :] = i_train_in[idx_sel_list[0], :, :, :]
    cnt_data = cnt_data + len_sel

    for nidx_d in range(1, len(idx_sel_list)):
        if (nidx_d + 1) % 20 == 0 or nidx_d == (len(idx_sel_list)-1):
            print(".")
        else:
            print(".", end='')

        data_read_tmp = np.load(filename2read[nidx_d], allow_pickle=False)
        i_train_in = data_read_tmp['data_i'] if is_npz else data_read_tmp[()]['data_i']
        i_train_in = i_train_in[:, :, :, idx_i_use]

        len_sel = len(idx_sel_list[nidx_d])
        idx_update_tmp = np.arange(cnt_data, cnt_data + len_sel)
        i_train[idx_update_tmp, :, :, :] = i_train_in[idx_sel_list[nidx_d], :, :, :]
        cnt_data = cnt_data + len_sel

    return i_train


def read_train_data_r(filename2read, idx_r_use, idx_sel_list, is_npz=False):
    """ Reads (stl-robustness) train-data. """
    data_size = 0
    for nidx_d in range(0, len(idx_sel_list)):
        data_size = data_size + len(idx_sel_list[nidx_d])

    # Read 1st-data
    filename2read_sel = filename2read[0]
    data_read_tmp = np.load(filename2read_sel, allow_pickle=True)
    r_train_in = data_read_tmp['data_r'] if is_npz else data_read_tmp[()]['data_r']
    r_train_in = r_train_in[:, idx_r_use]

    # Set r_train
    r_train = np.zeros((data_size, len(idx_r_use)), dtype=np.float32)
    cnt_data = 0

    len_sel = len(idx_sel_list[0])
    idx_update_tmp = np.arange(cnt_data, cnt_data + len_sel)
    r_train[idx_update_tmp, :, :, :] = r_train_in[idx_sel_list[0], :]
    cnt_data = cnt_data + len_sel

    for nidx_d in range(1, len(idx_sel_list)):
        data_read_tmp = np.load(filename2read[nidx_d], allow_pickle=False)
        r_train_in = data_read_tmp['data_r'] if is_npz else data_read_tmp[()]['data_r']
        r_train_in = r_train_in[:, idx_r_use]

        len_sel = len(idx_sel_list[nidx_d])
        idx_update_tmp = np.arange(cnt_data, cnt_data + len_sel)
        r_train[idx_update_tmp, :, :, :] = r_train_in[idx_sel_list[nidx_d], :]
        cnt_data = cnt_data + len_sel

    return r_train