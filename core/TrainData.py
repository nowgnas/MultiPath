# TRAIN DATA

import sys
sys.path.insert(0, "../")

import numpy as np
import math
from src.utils import *
from src.utils_sim import *
from src.utils_readdata import *


class TrainData(object):
    def __init__(self, filename2read_f, filename2read_i, filename2read_r, use_image, num_train, num_test,
                 h_prev, h_post, dim_p, idx_f_use, idx_i_use, idx_r_use, batch_size,
                 sp_x=True, sp_y=True, load_multi=False, num_near=6):

        self.filename2read_f = filename2read_f
        self.filename2read_i, self.filename2read_r = filename2read_i, filename2read_r

        self.use_image = use_image
        self.n_traindata, self.n_testdata = num_train, num_test

        self.dim_p = dim_p
        self.dim_i = []
        self.idx_f_use, self.idx_i_use, self.idx_r_use = idx_f_use, idx_i_use, idx_r_use
        self.batch_size = batch_size
        self.sp_x, self.sp_y = sp_x, sp_y
        self.load_multi = load_multi
        self.num_near = num_near

        self.h_prev = int(h_prev / 2) if sp_x else h_prev
        self.h_post = int(h_post / 2) if sp_y else h_post

        # Data
        self.xd_min, self.xd_max = [], []
        self.yd_min, self.yd_max = [], []
        self.f_min, self.f_max = [], []
        self.r_min, self.r_max = [], []

        self.x_train, self.f_train = [], []
        self.y0_train, self.y1_train, self.i_train, self.r_train = [], [], [], []
        self.x_train_n, self.f_train_n = [], []
        self.y0_train_n, self.y1_train_n, self.r_train_n = [], [], []  # Normalized target data (train)

        self.x_test, self.f_test = [], []
        self.y0_test, self.y1_test, self.i_test, self.r_test = [], [], [], []
        self.x_test_n, self.f_test_n = [], []
        self.y0_test_n, self.y1_test_n, self.r_test_n = [], [], []  # Normalized target data (test)

        self.idx_traindata, self.idx_testdata = [], []

        self.xd_train_mean, self.xd_train_std, self.yd_train_mean, self.yd_train_std = [], [], [], []
        self.x_train_mean, self.x_train_std = [], []
        self.y_train_mean, self.y_train_std = [], []
        self.f_train_mean, self.f_train_std, self.r_train_mean, self.r_train_std = [], [], [], []

        # Training
        self.n_batch_train, self.n_batch_test = 0, 0

        # Multiple case
        if self.load_multi:
            self.xnear_train, self.xnear_train_n = [], []
            self.y0near_train, self.y1near_train, self.y0near_train_n, self.y1near_train_n = [], [], [], []
            self.fnear_train, self.fnear_train_n = [], []
            self.rnear_train, self.rnear_train_n = [], []

            self.xnear_test, self.xnear_test_n = [], []
            self.y0near_test, self.y1near_test, self.y0near_test_n, self.y1near_test_n = [], [], [], []
            self.fnear_test, self.fnear_test_n = [], []
            self.rnear_test, self.rnear_test_n = [], []

    def processing(self):
        """ Processes data. """
        if self.load_multi:
            self.processing_multi()
        else:
            self.processing_single()

    def processing_single(self):
        """ Processes data (single). """
        data_size = int((self.n_traindata + self.n_testdata) * 1.2)
        print("Read feature-data")
        x_data, y0_data, y1_data, f_data, idx_sel_list = read_train_data_f(self.filename2read_f, self.dim_p,
                                                                           self.h_prev, self.h_post, self.idx_f_use,
                                                                           data_size, sp_x=self.sp_x,
                                                                           sp_y=self.sp_y, is_npz=True)

        if len(self.filename2read_i) > 0 and self.use_image == 1:
            print("Read image-data")
            i_data = read_train_data_i(self.filename2read_i, self.idx_i_use, idx_sel_list, is_npz=True)
            size_i = np.shape(i_data)
            self.dim_i = [size_i[1], size_i[2], size_i[3]]
        else:
            i_data = []

        if len(self.filename2read_r) > 0:
            print("Read robustness-data")
            r_data = read_train_data_r(self.filename2read_r, self.idx_r_use, idx_sel_list, is_npz=True)
        else:
            r_data = []

        self.processing_common(x_data, y0_data, y1_data, f_data, i_data, r_data)

    def processing_multi(self):
        """ Processes data (multiple). """
        pass

    def processing_common(self, x_data, y0_data, y1_data, f_data, i_data, r_data):
        """ Processes data (common). """
        x_data = x_data.astype(np.float32)
        y0_data = y0_data.astype(np.float32)
        y1_data = y1_data.astype(np.float32)

        x_data_r = np.reshape(x_data, (-1, self.dim_p))
        y1_data_r = np.reshape(y1_data, (-1, self.dim_p))
        self.xd_min, self.xd_max = np.amin(x_data_r, axis=0), np.amax(x_data_r, axis=0)
        self.yd_min, self.yd_max = np.amin(y1_data_r, axis=0), np.amax(y1_data_r, axis=0)
        if len(self.idx_f_use) > 0:
            self.f_min, self.f_max = np.amin(f_data, axis=0), np.amax(f_data, axis=0)
        if len(self.idx_r_use) > 0:
            self.r_min, self.r_max = np.amin(r_data, axis=0), np.amax(r_data, axis=0)

        n_data = x_data.shape[0]
        idx_data_random = np.random.permutation(n_data)

        # Set indexes for 'train' & 'test'
        self.idx_traindata = idx_data_random[np.arange(0, self.n_traindata)]
        self.idx_testdata = idx_data_random[np.arange(self.n_traindata, self.n_traindata + self.n_testdata)]

        # Set train-data
        self.x_train = x_data[self.idx_traindata, :]
        self.y0_train = y0_data[self.idx_traindata, :]
        self.y1_train = y1_data[self.idx_traindata, :]

        if len(self.idx_f_use) > 0:
            self.f_train = f_data[self.idx_traindata, :]

        if len(self.idx_i_use) > 0 and self.use_image == 1:
            self.i_train = i_data[self.idx_traindata, :, :]

        if len(self.idx_r_use) > 0:
            self.r_train = r_data[self.idx_traindata, :]

        # Set test-data
        self.x_test = x_data[self.idx_testdata, :]
        self.y0_test = y0_data[self.idx_testdata, :]
        self.y1_test = y1_data[self.idx_testdata, :]

        if len(self.idx_f_use) > 0:
            self.f_test = f_data[self.idx_testdata, :]

        if len(self.idx_i_use) > 0 and self.use_image == 1:
            self.i_test = i_data[self.idx_testdata, :]

        if len(self.idx_r_use) > 0:
            self.r_train = r_data[self.idx_testdata, :]

        # Data mean & scale
        x_train_r = np.reshape(self.x_train, (-1, self.dim_p))
        y_train_r = np.reshape(self.y1_train, (-1, self.dim_p))
        _, self.xd_train_mean, self.xd_train_std = normalize_data(x_train_r)
        _, self.yd_train_mean, self.yd_train_std = normalize_data(y_train_r)

        self.x_train_mean = np.tile(self.xd_train_mean, self.h_prev)
        self.x_train_std = np.tile(self.xd_train_std, self.h_prev)
        self.y_train_mean = np.tile(self.yd_train_mean, self.h_post)
        self.y_train_std = np.tile(self.yd_train_std, self.h_post)
        if len(self.idx_f_use) > 0:
            _, self.f_train_mean, self.f_train_std = normalize_data(self.f_train)
        if len(self.idx_r_use) > 0:
            _, self.r_train_mean, self.r_train_std = normalize_data(self.r_train)

        # Normalize
        self.x_train_n = normalize_data_wrt_mean_scale(self.x_train, self.x_train_mean, self.x_train_std)
        self.y0_train_n = normalize_data_wrt_mean_scale(self.y0_train, self.y_train_mean, self.y_train_std)
        self.y1_train_n = normalize_data_wrt_mean_scale(self.y1_train, self.y_train_mean, self.y_train_std)

        self.x_test_n = normalize_data_wrt_mean_scale(self.x_test, self.x_train_mean, self.x_train_std)
        self.y0_test_n = normalize_data_wrt_mean_scale(self.y0_test, self.y_train_mean, self.y_train_std)
        self.y1_test_n = normalize_data_wrt_mean_scale(self.y1_test, self.y_train_mean, self.y_train_std)

        if len(self.idx_f_use) > 0:
            self.f_train_n = normalize_data_wrt_mean_scale(self.f_train, self.f_train_mean, self.f_train_std)
            self.f_test_n = normalize_data_wrt_mean_scale(self.f_test, self.f_train_mean, self.f_train_std)

        if len(self.idx_r_use) > 0:
            self.r_train_n, self.r_train_mean, self.r_train_std = normalize_data(self.r_train)
            self.r_test_n = normalize_data_wrt_mean_scale(self.r_test, self.r_train_mean, self.r_train_std)

        # Reshape
        self.x_train = np.reshape(self.x_train, (self.n_traindata, self.h_prev, self.dim_p))
        self.y0_train = np.reshape(self.y0_train, (self.n_traindata, self.h_post, self.dim_p))
        self.y1_train = np.reshape(self.y1_train, (self.n_traindata, self.h_post, self.dim_p))
        self.x_train_n = np.reshape(self.x_train_n, (self.n_traindata, self.h_prev, self.dim_p))
        self.y0_train_n = np.reshape(self.y0_train_n, (self.n_traindata, self.h_post, self.dim_p))
        self.y1_train_n = np.reshape(self.y1_train_n, (self.n_traindata, self.h_post, self.dim_p))

        self.x_test = np.reshape(self.x_test, (self.n_testdata, self.h_prev, self.dim_p))
        self.y0_test = np.reshape(self.y0_test, (self.n_testdata, self.h_post, self.dim_p))
        self.y1_test = np.reshape(self.y1_test, (self.n_testdata, self.h_post, self.dim_p))
        self.x_test_n = np.reshape(self.x_test_n, (self.n_testdata, self.h_prev, self.dim_p))
        self.y0_test_n = np.reshape(self.y0_test_n, (self.n_testdata, self.h_post, self.dim_p))
        self.y1_test_n = np.reshape(self.y1_test_n, (self.n_testdata, self.h_post, self.dim_p))

        # Training size
        self.n_batch_train = math.floor(self.n_traindata / self.batch_size)
        self.n_batch_test = math.floor(self.n_testdata / self.batch_size)

    def get_batch(self, idx, is_train=True):
        if is_train:
            n_batchs = self.n_batch_train
        else:
            n_batchs = self.n_batch_test

        assert idx >= 0, "idx must be non negative"
        assert idx < n_batchs, "idx must be less than the number of batches:"
        start_idx = idx * self.batch_size
        indexes_sel = range(start_idx, start_idx + self.batch_size)

        return self.get_batch_from_indexes(indexes_sel, is_train=is_train)

    def get_batch_from_indexes(self, indexes, is_train=True):
        if is_train:
            x_data, f_data = self.x_train, self.f_train
            y0_data, y1_data = self.y0_train, self.y1_train
            i_data, r_data = self.i_train, self.r_train
            x_data_n, f_data_n = self.x_train_n, self.f_train_n
            y0_data_n, y1_data_n, r_data_n = self.y0_train_n, self.y1_train_n, self.r_train_n
        else:
            x_data, f_data = self.x_test, self.f_test
            y0_data, y1_data = self.y0_test, self.y1_test
            i_data, r_data = self.i_train, self.r_train
            x_data_n, f_data_n = self.x_test_n, self.f_test_n
            y0_data_n, y1_data_n, r_data_n = self.y0_test_n, self.y1_test_n, self.r_test_n

        x_batch = x_data[indexes, :, :]
        y0_batch = y0_data[indexes, :, :]
        y1_batch = y1_data[indexes, :, :]
        x_batch_n = x_data_n[indexes, :, :]
        y0_batch_n = y0_data_n[indexes, :, :]
        y1_batch_n = y1_data_n[indexes, :, :]

        # Feature
        if len(self.idx_f_use) > 0:
            f_batch = f_data[indexes, :]
            f_batch_n = f_data_n[indexes, :]
        else:
            f_batch, f_batch_n = [], []

        # Image
        if len(self.idx_i_use) > 0 and self.use_image == 1:
            i_batch = i_data[indexes, :, :, :]
        else:
            i_batch = []

        # Robustness (STL)
        if len(self.idx_r_use) > 0:
            r_batch = r_data[indexes, :]
            r_batch_n = r_data_n[indexes, :]
        else:
            r_batch, r_batch_n = [], []

        if not self.load_multi:
            dict_out = {'x_batch': x_batch, 'y0_batch': y0_batch, 'y1_batch': y1_batch,
                        'f_batch': f_batch, 'i_batch': i_batch, 'r_batch': r_batch,
                        'x_batch_n': x_batch_n, 'y0_batch_n': y0_batch_n, 'y1_batch_n': y1_batch_n,
                        'f_batch_n': f_batch_n, 'r_batch_n': r_batch_n}
        else:
            if is_train:
                xnear_data = self.xnear_train
                y0near_data, y1near_data = self.y0near_train, self.y1near_train
                fnear_data, rnear_data = self.fnear_train, self.rnear_train

                xnear_data_n = self.xnear_train_n
                y0near_data_n, y1near_data_n = self.y0near_train_n, self.y1near_train_n
                fnear_data_n, rnear_data_n = self.fnear_train_n, self.rnear_train_n
            else:
                xnear_data = self.xnear_test
                y0near_data, y1near_data = self.y0near_test, self.y1near_test
                fnear_data, rnear_data = self.fnear_test, self.rnear_test

                xnear_data_n = self.xnear_test_n
                y0near_data_n, y1near_data_n = self.y0near_test_n, self.y1near_test_n
                fnear_data_n, rnear_data_n = self.fnear_test_n, self.rnear_test_n

            xnear_batch = xnear_data[indexes, :, :, :]
            y0near_batch = y0near_data[indexes, :, :, :]
            y1near_batch = y1near_data[indexes, :, :, :]

            xnear_batch_n = xnear_data_n[indexes, :, :, :]
            y0near_batch_n = y0near_data_n[indexes, :, :, :]
            y1near_batch_n = y1near_data_n[indexes, :, :, :]

            if len(self.idx_f_use) > 0:  # Feature
                fnear_batch = fnear_data[indexes, :, :]
                fnear_batch_n = fnear_data_n[indexes, :, :]
            else:
                fnear_batch, fnear_batch_n = [], []

            if len(self.idx_r_use) > 0:  # Robustness (STL)
                rnear_batch = rnear_data[indexes, :, :]
                rnear_batch_n = rnear_data_n[indexes, :, :]
            else:
                rnear_batch, rnear_batch_n = [], []

            dict_out = {'x_batch': x_batch, 'y0_batch': y0_batch, 'y1_batch': y1_batch,
                        'f_batch': f_batch, 'i_batch': i_batch, 'r_batch': r_batch,
                        'x_batch_n': x_batch_n, 'y0_batch_n': y0_batch_n, 'y1_batch_n': y1_batch_n,
                        'f_batch_n': f_batch_n, 'r_batch_n': r_batch_n,
                        'xnear_batch': xnear_batch, 'y0near_batch': y0near_batch, 'y1near_batch': y1near_batch,
                        'xnear_batch_n': xnear_batch_n, 'y0near_batch_n': y0near_batch_n,
                        'y1near_batch_n': y1near_batch_n,
                        'fnear_batch': fnear_batch, 'fnear_batch_n': fnear_batch_n,
                        'rnear_batch': rnear_batch, 'rnear_batch_n': rnear_batch_n}

        return dict_out

    def shuffle_traindata(self):
        idx_random = np.random.permutation(self.n_traindata)
        self.x_train = self.x_train[idx_random, :, :]
        self.y0_train = self.y0_train[idx_random, :, :]
        self.y1_train = self.y1_train[idx_random, :, :]

        self.x_train_n = self.x_train_n[idx_random, :, :]
        self.y0_train_n = self.y0_train_n[idx_random, :, :]
        self.y1_train_n = self.y1_train_n[idx_random, :, :]

        if len(self.idx_f_use) > 0:
            self.f_train = self.f_train[idx_random, :]
            self.f_train_n = self.f_train_n[idx_random, :]

        if len(self.idx_i_use) > 0 and self.use_image == 1:
            self.i_train = self.i_train[idx_random, :]

        if len(self.idx_r_use) > 0:
            self.r_train = self.r_train[idx_random, :]
            self.r_train_n = self.r_train_n[idx_random, :]

        if self.load_multi:
            self.xnear_train = self.xnear_train[idx_random, :, :, :]
            self.y0near_train = self.y0near_train[idx_random, :, :, :]
            self.y1near_train = self.y1near_train[idx_random, :, :, :]
            self.xnear_train_n = self.xnear_train_n[idx_random, :, :, :]
            self.y0near_train_n = self.y0near_train_n[idx_random, :, :, :]
            self.y1near_train_n = self.y1near_train_n[idx_random, :, :, :]

            if len(self.idx_f_use) > 0:
                self.fnear_train = self.fnear_train[idx_random, :, :]
                self.fnear_train_n = self.fnear_train_n[idx_random, :, :]

            if len(self.idx_r_use) > 0:
                self.rnear_train = self.rnear_train[idx_random, :, :]
                self.rnear_train_n = self.rnear_train_n[idx_random, :, :]
