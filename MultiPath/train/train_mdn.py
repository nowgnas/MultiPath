# TRAIN: PREDICTION WITH MIXTURE DENSITY NETWORK (MDN)
#
#   - INPUT: current-feature, previous trajectory
#   - TARGET: posterior trajectory


from __future__ import print_function, division, absolute_import

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Select GPU driver

import sys
sys.path.insert(0, "../../")

import argparse
import pandas as pd
import time

from core.TrainData import *
from MultiPath.model.Mdn import *
from tqdm import trange
# from tqdm import tqdm


def main(_args, params):
    txt_type = 'i' if params['use_image'] == 1 else 'f'

    model_name = "mdn_i{:d}o{:d}_sp{:d}{:d}_n{:d}_{:s}".format(
        params['len_x'], params['len_y'], params['sp_x'], params['sp_y'], params['n_component_gmm'], txt_type)

    directory2save = "../trained_model/{:s}".format("mdn")
    if not os.path.exists(directory2save):
        os.makedirs(directory2save)

    # Time-steps of input (horizon)
    h_prev, h_post = params['len_x'], params['len_y']

    # STEP 1: LOAD DATA -----------------------------------------------------------------------------------------------#
    directory2load = "../../data/train"
    filename2read_f, filename2read_i, filename2read_r = [], [], []

    trackid_exclude = np.arange(5, 61, 5, dtype=np.int32)
    trackid_include = np.setdiff1d(np.arange(24, 61), trackid_exclude)
    for nidx_d in range(0, trackid_include.shape[0]):
        f_data_filename = "{:s}/train_f_highd_{:d}.npz".format(directory2load, trackid_include[nidx_d])
        filename2read_f.append(f_data_filename)

        i_data_filename = "{:s}/train_i_highd_{:d}.npz".format(directory2load, trackid_include[nidx_d])
        filename2read_i.append(i_data_filename)

    sp_x, sp_y = (params['sp_x'] == 1), (params['sp_y'] == 1)
    traindata = TrainData(filename2read_f, filename2read_i, filename2read_r, params['use_image'],
                          params['num_train'], params['num_test'], h_prev, h_post, params['dim_p'],
                          params['idx_f_use'], params['idx_i_use'], [], params['batch_size'], sp_x=sp_x, sp_y=sp_y,
                          load_multi=False)
    traindata.processing()

    # STEP 2: SAVE PARAMETERS -----------------------------------------------------------------------------------------#
    # Update mean & std
    params.update({'x_mean': traindata.x_train_mean.tolist(), 'x_std': traindata.x_train_std.tolist(),
                   'y_mean': traindata.y_train_mean.tolist(), 'y_std': traindata.y_train_std.tolist(),
                   'f_mean': traindata.f_train_mean.tolist(), 'f_std': traindata.f_train_std.tolist(),
                   'xd_mean': traindata.xd_train_mean.tolist(), 'xd_std': traindata.xd_train_std.tolist(),
                   'yd_mean': traindata.yd_train_mean.tolist(), 'yd_std': traindata.yd_train_std.tolist(),
                   'n_batch': traindata.n_batch_train, 'dim_i': traindata.dim_i})

    filename2save0_ = "p_{:s}".format(model_name)
    filename2save0 = directory2save + '/' + filename2save0_
    import json
    json = json.dumps(params)
    f = open(filename2save0, "w")
    f.write(json)
    f.close()

    # STEP 3: DEFINE NETWORK ------------------------------------------------------------------------------------------#
    # Add support for dot access for auxiliary function use
    hps_model = DotDict(params)

    # Build and compile model
    model_pred = Mdn(hps=hps_model)
    time.sleep(0.5)

    # STEP 4: TRAIN ---------------------------------------------------------------------------------------------------#
    # A pandas dataframe to save the loss information to
    losses_train = pd.DataFrame(columns=['loss'])  # (train) loss to save
    losses_valid = pd.DataFrame(columns=['loss'])  # (valid) loss to save
    filename_losstrain = "l_train_" + model_name + ".csv"
    filename_lossvalid = "l_valid_" + model_name + ".csv"

    print_w, print_prec = 7, 3

    # Run loop
    step = 0.0
    for epoch in range(hps_model.epochs):
        # Train
        loss_train = []
        t_train = trange(traindata.n_batch_train, leave=False)
        for nidx_i in t_train:
            dict_batch = traindata.get_batch(nidx_i, is_train=True)
            x_batch_n = tf.convert_to_tensor(dict_batch['x_batch_n'], dtype=tf.float32)
            y1_batch_n = tf.convert_to_tensor(dict_batch['y1_batch_n'], dtype=tf.float32)
            f_batch_n = tf.convert_to_tensor(dict_batch['f_batch_n'], dtype=tf.float32)
            i_batch = tf.convert_to_tensor(dict_batch['i_batch'], dtype=tf.float32)

            lr = model_pred.lr_decayed(step)

            l_t = model_pred.train(x_batch_n, f_batch_n, i_batch, y1_batch_n)
            step = step + 1
            loss_train.append(keras.backend.eval(l_t))

            t_txt = "[TRAIN, epoch: {:>2d}] lr: {:1.4f} loss: {:{width}.{prec}f}".\
                format(epoch + 1, lr, l_t, width=print_w, prec=print_prec)

            t_train.set_description(t_txt)
            # t_train.refresh()  # to show immediately the update

        losses_train.loc[len(losses_train)] = np.mean(loss_train, axis=0)
        time.sleep(0.2)

        # Valid
        loss_valid = []
        t_valid = trange(traindata.n_batch_test, leave=False)
        for nidx_i in t_valid:
            dict_batch = traindata.get_batch(nidx_i, is_train=False)
            x_batch_n = tf.convert_to_tensor(dict_batch['x_batch_n'], dtype=tf.float32)
            y1_batch_n = tf.convert_to_tensor(dict_batch['y1_batch_n'], dtype=tf.float32)
            f_batch_n = tf.convert_to_tensor(dict_batch['f_batch_n'], dtype=tf.float32)
            i_batch = tf.convert_to_tensor(dict_batch['i_batch'], dtype=tf.float32)

            l_v = model_pred.compute_loss(x_batch_n, f_batch_n, i_batch, y1_batch_n)
            loss_valid.append(keras.backend.eval(l_v))
            # t_valid.refresh()  # to show immediately the update
        losses_valid.loc[len(losses_valid)] = np.mean(loss_valid, axis=0)
        time.sleep(0.2)

        # Print
        print("[epoch: {:>2d}] loss: {:{width}.{prec}f}, valid-loss: {:{width}.{prec}f}".
              format(epoch + 1, np.mean(loss_train), np.mean(loss_valid), width=print_w, prec=print_prec))

        # Save
        if (epoch + 1) % 10 == 0:
            fileanme2save1_ = "{:s}_e{:d}".format(model_name, int(epoch + 1))
            fileanme2save1 = directory2save + '/' + fileanme2save1_
            model_pred.save_trained_weights(fileanme2save1)
            time.sleep(0.2)

        # Shuffle train-data
        traindata.shuffle_traindata()

    # Save training result
    losses_train.to_csv(directory2save + "/" + filename_losstrain)
    losses_valid.to_csv(directory2save + "/" + filename_lossvalid)


if __name__ == "__main__":
    # Parse arguments and use defaults when needed
    parser = argparse.ArgumentParser(description='Main script for training MDN')

    # Data params:
    parser.add_argument('--len_x', type=int, default=6, help='Sequence length (previous-trajectory)')
    parser.add_argument('--len_y', type=int, default=18, help='Sequence length (posterior-trajectory)')
    parser.add_argument('--sp_x', type=int, default=0, help='Use sparse x (previous-trajectory)')
    parser.add_argument('--sp_y', type=int, default=0, help='Use sparse y (posterior-trajectory)')

    # Experiment params:
    parser.add_argument('--use_image', type=int, default=0, help='Whether to use image.')
    parser.add_argument('--n_component_gmm', type=int, default=12, help='Number of mixture components.')
    parser.add_argument('--batch_size', type=int, default=64, help='Minibatch size. (32, 64, 128, 256)')

    args = parser.parse_args()

    # Update parameters
    hparams = get_default_hparams()

    # Set indexes to use (feature)
    idx_f_use = [3, 4, 6, 7, 8, 9, 10, 11]
    dim_f = len(idx_f_use)

    # Set indexes to use (image)
    idx_i_use = [0, 1, 2, 3]

    hparams.update({'len_x': args.len_x, 'len_y': args.len_y, 'sp_x': args.sp_x, 'sp_y': args.sp_y,
                    'dim_f': dim_f, 'idx_f_use': idx_f_use, 'idx_i_use': idx_i_use,
                    'use_image': args.use_image, 'n_component_gmm': args.n_component_gmm,
                    'batch_size': args.batch_size})

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    main(args, hparams)
