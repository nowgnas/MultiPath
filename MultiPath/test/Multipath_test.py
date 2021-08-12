# PREDICTION WITH MIXTURE DENSITY NETWORK (MDN)
#
#   - INPUT: current-feature, previous trajectory
#   - TARGET: posterior trajectory

from __future__ import print_function

import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Select GPU driver

import sys

sys.path.insert(0, "../../")

import argparse
import os
import random

from core.SimTrack import *
from src.utils import *
from src.utils_sim import *
from MultiPath.model.multipath import *

import matplotlib.pyplot as plt


def main(params):
    directory_model = "../trained_model"
    directory_track = "../../data/track"
    directory_vehicle = "../../data/vehicle"
    directory_result = "../result"

    # SET KEY PARAMETERS ----------------------------------------------------------------------------------------------#
    SAVE_RESULT = params['SAVE_RESULT']

    # Environment parameter (track-name, id of ego-vehicle)
    trackname, id_tv, idx_id_tv = params['trackname'], params['v_id'], params['v_id_ord']

    # Data parameter
    dim_p, delta_t = 2, 2
    sp_x, sp_y = params['sp_x'], params['sp_y']
    h_prev, h_post = params['len_x'], params['len_y']
    use_image = params['use_image']

    # MDN parameter
    n_component_gmm = params['n_component_gmm']
    epoch = params['epoch']

    # Number of sampled trajectory
    n_sample = params['n_sample']

    # SET PARAMETERS --------------------------------------------------------------------------------------------------#
    rx_tv, ry_tv = 4.2, 1.9  # Size of target-vehicle

    h_prev_sp, h_post_sp = int(h_prev / delta_t), int(h_post / delta_t)
    h_post_model = h_post_sp if sp_y == 1 else h_post

    # Set model name
    txt_type = 'i' if use_image == 1 else 'f'

    model_name = "multipath_i{:d}o{:d}_sp{:d}{:d}_n{:d}_{:s}".format(h_prev, h_post, sp_x, sp_y, n_component_gmm,
                                                                     txt_type)

    # LOAD MODEL PARAMETERS -------------------------------------------------------------------------------------------#
    directory2load = "{:s}/multipath".format(directory_model)
    param_filename = "p_" + model_name
    model_filename = model_name + "_e{:d}".format(epoch)

    import json
    with open(directory2load + "/" + param_filename) as json_file:
        hparams = json.load(json_file)

    # RECOVER MODEL ---------------------------------------------------------------------------------------------------#
    # Add support for dot access for auxiliary function use
    hps_model = DotDict(hparams)

    idx_f_use = hps_model.idx_f_use
    x_mean, x_std = np.array(hps_model.x_mean, dtype=np.float32), np.array(hps_model.x_std, dtype=np.float32)
    y_mean, y_std = np.array(hps_model.y_mean, dtype=np.float32), np.array(hps_model.y_std, dtype=np.float32)
    f_mean, f_std = np.array(hps_model.f_mean, dtype=np.float32), np.array(hps_model.f_std, dtype=np.float32)

    # Build model
    nn_pred = Mdn(hps=hps_model)
    nn_pred.load_trained_weights(directory2load + "/" + model_filename)

    # SET TEST ENVIRONMENT --------------------------------------------------------------------------------------------#
    # SET TRACK
    sim_track = SimTrack(trackname, directory_track)

    # LOAD VEHICLE DATA
    #       structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    trackname_split = trackname.split("_")
    trackname_track, trackname_id = trackname_split[0], trackname_split[1]
    if trackname_track == "us101" or trackname_track == "i80":
        vehicle_data_filename = "{:s}/dv_ngsim_{:s}_{:s}.npz".format(directory_vehicle, trackname_track.lower(),
                                                                     trackname_id)
        is_track_simple = 0
    elif "highd" in trackname_track:
        vehicle_data_filename = "{:s}/dv_highD_{:s}.npz".format(directory_vehicle, trackname_id)
        is_track_simple = 1
    else:
        vehicle_data_filename = "{:s}/dv_ngsim_us101_1.npz".format(directory_vehicle)
        is_track_simple = 0

    data_in = np.load(vehicle_data_filename)
    data_v = data_in['data']

    # Set target vehicle id
    if id_tv == -1:
        id_unique = np.unique(data_v[:, -1])
        if not (id_tv in id_unique):
            if (0 <= idx_id_tv) and (idx_id_tv <= len(id_unique)):
                id_tv = id_unique[idx_id_tv]
            else:
                print("[WARNING] CANNOT FIND TARGET-ID!")
                idx_tv_new = random.randint(0, len(id_unique))
                id_tv = id_unique[idx_tv_new]

    # SET INIT-STATE
    t_init, t_horizon, data_tv_init, data_tv, data_ov = set_initstate_pred(id_tv, ry_tv, rx_tv, data_v, sim_track,
                                                                           is_track_simple=0)
    t_min, t_max = t_init, t_init + t_horizon
    print("[SET-ENV] track-name: {:s}, vehicle-id: {:d}, t_min: {:d}, t_max: {:d}, t_len: {:d}".
          format(trackname, id_tv, t_min, t_max, t_max - t_min + 1))

    # SET DIRECTORY ---------------------------------------------------------------------------------------------------#
    if SAVE_RESULT >= 1:
        directory2save = "{:s}/{:s}/pred_{:s}_id{:d}".format(directory_result, model_name, trackname, id_tv)
        if not os.path.exists(directory2save):
            os.makedirs(directory2save)

    # MAKE PREDICTION -------------------------------------------------------------------------------------------------#
    # Set output repository
    data_tv_t_list, data_ov_t_list, s_tv_t_list = [], [], []
    id_near_t_list, id_rest_t_list = [], []

    traj_ov_near_list, size_ov_near_list = [], []

    traj_tv = np.zeros((t_horizon + 1, 3), dtype=np.float32)
    y_tv_list, y_tv_pred_list = [], []
    y_tv_pred_mu_list, y_tv_pred_pi_list = [], []

    if use_image == 1:
        idx_i_use = hps_model.idx_i_use
        img_width, img_height = 50, 50
        img_dim = [hps_model.dim_i[0], hps_model.dim_i[1]]
        img_dist = np.sqrt(img_width ** 2 + img_height ** 2) / 2
        img_numov = [6, 10]
        do_first_plot = False

    print("[SIMULATION START]--------------------------------------------------")
    for t_cur in range(t_init, t_init + t_horizon + 1):
        if (t_cur - t_init) % 10 == 0:
            print(".", end='')

        # STEP1: GET CURRENT INFO -------------------------------------------------------------------------------------#
        data_tv_t, idx_tv_t = select_data_t(data_tv, t_cur)
        s_tv_t = data_tv_t[1:5]
        data_ov_t, idx_ov_t = select_data_t(data_ov, t_cur)
        segidx, laneidx = int(data_tv_t[7]), int(data_tv_t[8])
        num_lane = sim_track.num_lane[segidx]

        # Get feature
        dict_info_t = get_info_t(t_cur, id_tv, data_tv, data_tv_t, data_ov_t, idx_f_use, h_prev, h_post,
                                 [], [], 2, sim_track, use_intp=0, f_precise=False)
        f_t, lanewidth_t = dict_info_t['f'], dict_info_t['lanewidth']
        id_near_t, id_rest_t = dict_info_t['id_near'], dict_info_t['id_rest']
        y_tv = dict_info_t['y_ev']
        xenc_ev, xenc_sp_ev = dict_info_t['x_ev_enc'], dict_info_t['x_ev_enc_sp']
        yenc_ev, yenc_sp_ev = dict_info_t['y_ev_enc'], dict_info_t['y_ev_enc_sp']

        # Get near other vehicles
        data_ov_near_t, data_ov_near_t_list = select_data_near(data_ov, id_near_t)
        _, traj_ov_near_sq, size_ov_near_sq = get_vehicle_traj(data_ov_near_t, t_cur, h_post, handle_remain=1)

        # Get image
        if use_image == 1:
            data_ov_t_i = select_data_dist(data_ov_t, data_tv_t[1:3], img_dist)
            if not do_first_plot:
                plt.close('all')
                plt.pause(0.1)
                f_img, img_fig, img_ax, img_t_poly_segs, img_t_line_o_segs, img_t_line_i_segs, img_ev_seg, \
                img_ov1_segs, img_ov2_segs = get_feature_image_init(sim_track, data_tv_t, data_ov_t_i, id_near_t,
                                                                    trackname, laneidx, num_lane, img_width,
                                                                    img_height, img_dim,
                                                                    num1=img_numov[0], num2=img_numov[1])
                do_first_plot = True
            else:
                f_img, img_fig, img_ax, img_t_poly_segs, img_t_line_o_segs, img_t_line_i_segs, img_ev_seg, \
                img_ov1_segs, img_ov2_segs = \
                    get_feature_image_update(img_fig, img_ax, img_t_poly_segs, img_t_line_o_segs,
                                             img_t_line_i_segs, img_ev_seg, img_ov1_segs, img_ov2_segs,
                                             sim_track, data_tv_t, data_ov_t_i, id_near_t, trackname, laneidx,
                                             num_lane, img_dim, num1=img_numov[0], num2=img_numov[1])

        # STEP2: RUN MODEL --------------------------------------------------------------------------------------------#
        # x_input
        if sp_x == 1:
            xenc_sp_ev = normalize_data_wrt_mean_scale(xenc_sp_ev, x_mean, x_std)
        else:
            xenc_ev = normalize_data_wrt_mean_scale(xenc_ev, x_mean, x_std)
        x_input = xenc_sp_ev.reshape((1, h_prev_sp, dim_p)) if sp_x == 1 else xenc_ev.reshape((1, h_prev, dim_p))

        # y input
        if sp_y == 1:
            yenc_sp_ev = normalize_data_wrt_mean_scale(yenc_sp_ev, y_mean, y_std)
        else:
            yenc_ev = normalize_data_wrt_mean_scale(yenc_ev, y_mean, y_std)
        y_input = yenc_sp_ev.reshape((1, 18, dim_p)) if sp_y == 1 else yenc_ev.reshape((1, 18, dim_p))

        # f_input or i_input
        if use_image == 0:
            f_t = normalize_data_wrt_mean_scale(f_t, f_mean, f_std)  # Normalize feature
            f_input = f_t.reshape((1, -1))
            i_input = []
        else:
            f_img = np.expand_dims(f_img, axis=0)
            f_img = f_img.astype('float32')
            i_input = f_img[:, :, :, idx_i_use]
            f_input = []

        y_tv_pred = []
        valpred_y_list = nn_pred.sample(x_input, f_input, i_input, y_input, num_sample=n_sample)
        for nidx_s in range(0, n_sample):
            valpred_y = valpred_y_list[nidx_s]
            _y_pred = recover_traj_pred(s_tv_t, valpred_y, y_mean, y_std, dim_p, h_post_model, sim_track,
                                        is_track_simple=is_track_simple, xth=0.33 * lanewidth_t, yth=0.33 * lanewidth_t)

            y_pred = interpolate_traj(_y_pred, alpha=delta_t) if sp_y == 1 else _y_pred
            y_pred = update_heading_traj(y_pred, theta_thres=0.1 * np.pi)  # Update heading

            y_tv_pred.append(y_pred)

        y_tv_pred_mu, y_tv_pred_pi = [], []
        _, _y_mu, _, _y_log_pi = nn_pred.get_mdn_gmmdiag(x_input, f_input, i_input, y_input)
        _y_mu, _y_log_pi = _y_mu.numpy(), _y_log_pi.numpy()
        for nidx_i in range(0, 1):
            _y_mu_sel = _y_mu[:, :, nidx_i]
            _y_log_pi_sel = np.exp(_y_log_pi[0, nidx_i])

            _y_mu_sel = recover_traj_pred(s_tv_t, _y_mu_sel, y_mean, y_std, dim_p, h_post_model, sim_track,
                                          is_track_simple=is_track_simple, xth=0.33 * lanewidth_t,
                                          yth=0.33 * lanewidth_t)

            _y_mu_sel = interpolate_traj(_y_mu_sel, alpha=delta_t) if sp_y == 1 else _y_mu_sel
            _y_mu_sel = update_heading_traj(_y_mu_sel, theta_thres=0.1 * np.pi)  # Update heading

            y_tv_pred_mu.append(_y_mu_sel)
            y_tv_pred_pi.append(_y_log_pi_sel)

        # STEP3: UPDATE REPOSITORY ------------------------------------------------------------------------------------#
        data_tv_t_list.append(data_tv_t)
        data_ov_t_list.append(data_ov_t)
        s_tv_t_list.append(s_tv_t)
        id_near_t_list.append(id_near_t)
        id_rest_t_list.append(id_rest_t)

        traj_ov_near_list.append(traj_ov_near_sq)
        size_ov_near_list.append(size_ov_near_sq)

        traj_tv[t_cur - t_init, :] = s_tv_t[0:3]
        y_tv_list.append(y_tv)
        y_tv_pred_list.append(y_tv_pred)

        y_tv_pred_mu_list.append(y_tv_pred_mu)
        y_tv_pred_pi_list.append(y_tv_pred_pi)

    print("")
    print("[SIMULATION END]--------------------------------------------------")

    # SAVE ------------------------------------------------------------------------------------------------------------#
    if SAVE_RESULT >= 1:
        filename2save_result = "{:s}/result.npz".format(directory2save)
        np.savez_compressed(filename2save_result, data_tv_t_list=data_tv_t_list, data_ov_t_list=data_ov_t_list,
                            s_tv_t_list=s_tv_t_list, id_near_t_list=id_near_t_list, id_rest_t_list=id_rest_t_list,
                            traj_ov_near_list=traj_ov_near_list, size_ov_near_list=size_ov_near_list, traj_tv=traj_tv,
                            y_tv_list=y_tv_list, y_tv_pred_list=y_tv_pred_list, y_tv_pred_mu_list=y_tv_pred_mu_list,
                            y_tv_pred_pi_list=y_tv_pred_pi_list)
        print("[SAVED]")
    print("[DONE]")


if __name__ == "__main__":
    # Parse arguments and use defaults when needed
    parser = argparse.ArgumentParser(description='Main script for testing MDN')
    parser.add_argument('--SAVE_RESULT', type=int, default=1, help='0: none, 1: save result')
    # Data params:
    parser.add_argument('--len_x', type=int, default=6, help='Sequence length (previous-trajectory)')
    parser.add_argument('--len_y', type=int, default=18, help='Sequence length (posterior-trajectory)')
    parser.add_argument('--sp_x', type=int, default=0, help='Use sparse x.')
    parser.add_argument('--sp_y', type=int, default=0, help='Use sparse y.')
    # Experiment params:
    parser.add_argument('--use_image', type=int, default=0, help='Whether to use image.')
    parser.add_argument('--n_component_gmm', type=int, default=12, help='Number of mixture components.')
    parser.add_argument('--n_sample', type=int, default=25, help='Number of sampled trajectories')

    parser.add_argument('--trackname', type=str, default='highd_25', help='Track name: us101_1 ~ us101_3, '
                                                                          'i80_1 ~ i80_3, highd_1 ~ highd_60')
    parser.add_argument('--v_id', type=int, default=68, help='vehicle-id')
    parser.add_argument('--v_id_ord', type=int, default=-1, help='vehicle-id order')
    args = parser.parse_args()

    key_params = {'SAVE_RESULT': args.SAVE_RESULT,
                  'len_x': args.len_x, 'len_y': args.len_y, 'sp_x': args.sp_x, 'sp_y': args.sp_y,
                  'use_image': args.use_image, 'n_component_gmm': args.n_component_gmm,
                  'epoch': 50, 'n_sample': args.n_sample,
                  'trackname': args.trackname, 'v_id': args.v_id, 'v_id_ord': args.v_id_ord}

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

    main(key_params)
