# DRAW PREDICTION-RESULT (MIXTURE DENSITY NETWORK (MDN))
#   : Screen with matplotlib

from __future__ import print_function

import sys

sys.path.insert(0, "../../")

import argparse
import os
import time

from core.SimScreenMatplotlib import SimScreenMatplotlib
from core.SimTrack import *
from src.get_rgb import get_rgb
from src.utils import *
from src.utils_sim import *

import matplotlib
import matplotlib.pyplot as plt


def main(params):
    directory_track = "../../data/track"
    directory_read = "../result"
    directory_write = "../result"

    # SET KEY PARAMETERS ----------------------------------------------------------------------------------------------#
    MODE_SCREEN, PLOT_BK = params['MODE_SCREEN'], params['PLOT_BK']

    # Test environment parameter
    trackname, id_tv = params['trackname'], params['v_id']

    # Model-name to load
    model_name = params['model_name']

    # Set directory to load
    directory2read = "{:s}/{:s}/pred_{:s}_id{:d}".format(directory_read, model_name, trackname, id_tv)

    # LOAD RESULT -----------------------------------------------------------------------------------------------------#
    r_filename = directory2read + '/result.npz'

    data_read_tmp = np.load(r_filename, allow_pickle=True)
    data_tv_t_list = data_read_tmp["data_tv_t_list"]
    data_ov_t_list = data_read_tmp["data_ov_t_list"]
    s_tv_t_list = data_read_tmp["s_tv_t_list"]
    id_near_t_list = data_read_tmp["id_near_t_list"]
    id_rest_t_list = data_read_tmp["id_rest_t_list"]
    traj_ov_near_list = data_read_tmp["traj_ov_near_list"]
    size_ov_near_list = data_read_tmp["size_ov_near_list"]
    traj_tv = data_read_tmp["traj_tv"]
    y_tv_list = data_read_tmp["y_tv_list"]
    y_tv_pred_list = data_read_tmp["y_tv_pred_list"]

    y_tv_pred_mu_list = data_read_tmp["y_tv_pred_mu_list"]
    y_tv_pred_pi_list = data_read_tmp["y_tv_pred_pi_list"]

    len_runtime = len(data_tv_t_list)
    print("length: {:d}".format(len_runtime))

    # Compute loss
    loss_y_tv_list_sort, y_tv_pred_list_sort = [], []
    for nidx_t in range(0, len_runtime):
        y_tv_t = y_tv_list[nidx_t]
        y_tv_pred_t = y_tv_pred_list[nidx_t]

        loss_y_tv_t = []
        for nidx_s in range(0, len(y_tv_pred_t)):
            y_tv_pred_t_s = y_tv_pred_t[nidx_s]
            _, min_dist_tmp = get_closest_pnt(y_tv_pred_t_s[-1, 0:2], y_tv_t[:, 0:2])
            loss_y_tv_t.append(min_dist_tmp)

        idx_sorted_tmp = np.argsort(loss_y_tv_t)
        idx_sorted_tmp = np.flip(idx_sorted_tmp, axis=0)
        idx_sorted_tmp = idx_sorted_tmp.astype(int)

        loss_y_tv_t_sort = [loss_y_tv_t[idx_tmp] for idx_tmp in idx_sorted_tmp]
        loss_y_tv_list_sort.append(loss_y_tv_t_sort)

        y_tv_pred_t_sort = [y_tv_pred_t[idx_tmp] for idx_tmp in idx_sorted_tmp]
        y_tv_pred_list_sort.append(y_tv_pred_t_sort)

    # SET TEST ENVIRONMENT --------------------------------------------------------------------------------------------#
    # Set directory pic
    directory2save = "{:s}/{:s}/pred_{:s}_id{:d}".format(directory_write, model_name, trackname, id_tv)
    directory2save_pic = "{:s}/pic".format(directory2save)
    if not os.path.exists(directory2save_pic):
        os.makedirs(directory2save_pic)

    # Set track
    sim_track = SimTrack(trackname, directory_track)

    # Set screen
    sim_screen_m = SimScreenMatplotlib(MODE_SCREEN, PLOT_BK)
    sim_screen_m.set_pnts_track_init(sim_track.pnts_poly_track, sim_track.pnts_outer_border_track,
                                     sim_track.pnts_inner_border_track)

    if MODE_SCREEN == 1:
        height2width_ratio = 1.6 / 3.0
        screen_width = 7.0 * 2
        screen_xlen = 60
        dpi_save = 200 * 1
    else:
        track_xlen = sim_track.pnt_max[0] - sim_track.pnt_min[0]
        track_ylen = sim_track.pnt_max[1] - sim_track.pnt_min[1]
        height2width_ratio = track_ylen / track_xlen
        screen_width = 12.0
        dpi_save = 600

    # DRAW ------------------------------------------------------------------------------------------------------------#
    print("[DRAW START]--------------------------------------------------")
    for idx_t in range(0, len_runtime):
        if idx_t % 10 == 0:
            print(".", end='')

        data_tv_t_sel, data_ov_t_sel = data_tv_t_list[idx_t], data_ov_t_list[idx_t]
        s_tv_t_sel = s_tv_t_list[idx_t]
        id_near_t_sel, id_rest_t_sel = id_near_t_list[idx_t], id_rest_t_list[idx_t]
        traj_ov_near_sel, size_ov_near_sel = traj_ov_near_list[idx_t], size_ov_near_list[idx_t]
        y_tv_sel, y_tv_pred_sel = y_tv_list[idx_t], y_tv_pred_list[idx_t]
        loss_y_tv_sel_sort, y_tv_pred_sel_sort = loss_y_tv_list_sort[idx_t], y_tv_pred_list_sort[idx_t]
        traj_tv_hist = traj_tv[range(0, idx_t + 1), :]

        y_tv_pred_mu_sel = y_tv_pred_mu_list[idx_t]
        y_tv_pred_pi_sel = y_tv_pred_pi_list[idx_t]

        sim_screen_m.set_figure(screen_width, screen_width * height2width_ratio)
        if MODE_SCREEN == 1:
            sim_screen_m.set_pnt_range([s_tv_t_sel[0], s_tv_t_sel[1]], [screen_xlen, screen_xlen * height2width_ratio])

        # Draw track
        sim_screen_m.draw_track()

        if MODE_SCREEN == 1:
            sim_screen_m.draw_pred_basic(data_tv_t_sel, data_ov_t_sel, id_near_t_sel, id_rest_t_sel, y_tv_sel,
                                         y_tv_pred_sel_sort, traj_ov_near_sel, size_ov_near_sel, traj_tv_hist,
                                         loss_y_tv_sel_sort, draw_trajov=False, hcolormap="viridis")

            cmap_mu = matplotlib.cm.get_cmap('jet')
            for nidx_i in range(len(y_tv_pred_mu_sel)):
                _y_tv_pred_mu_sel = y_tv_pred_mu_sel[nidx_i]
                _y_tv_pred_pi_sel = y_tv_pred_pi_sel[nidx_i]
                _cmap_mu = cmap_mu(_y_tv_pred_pi_sel)
                _cmap_mu = _cmap_mu[0:3]
                sim_screen_m.draw_traj(_y_tv_pred_mu_sel[:, 0:2], 1.0, '--', _cmap_mu, op=1.0, zorder=3)
            sim_screen_m.update_view_range()
        else:
            sim_screen_m.draw_basic(data_tv_t_sel, data_ov_t_sel, id_near_t_sel, id_rest_t_sel, traj_tv_hist)

        plt.show(block=False)
        time.sleep(0.5)

        # SAVE-PIC
        plt.axis('off')
        filename2save_pic = "{:s}/m{:d}b{:d}_{:d}.png".format(directory2save_pic, MODE_SCREEN, PLOT_BK, idx_t)
        sim_screen_m.save_figure(filename2save_pic, dpi=dpi_save)
        time.sleep(0.5)
        plt.close('all')

    print("")
    print("[DRAW END]--------------------------------------------------")
    print("[DONE]--------------------------------------------------")


if __name__ == "__main__":

    # Parse arguments and use defaults when needed
    parser = argparse.ArgumentParser(description='Main script for drawing prediction.')
    parser.add_argument('--MODE_SCREEN', type=int, default=1, help='0: plot all track // 1: plot part of track')
    parser.add_argument('--PLOT_BK', type=int, default=0, help='0: White // 1: Black')
    parser.add_argument('--model_name', type=str, default='multipath_i6o18_sp00_n12_f', help='model name')
    parser.add_argument('--trackname', type=str, default='highd_25', help='Track name: us101_1 ~ us101_3, '
                                                                          'i80_1 ~ i80_3, highd_1 ~ highd_60')
    parser.add_argument('--v_id', type=int, default=68, help='vehicle-id')
    args = parser.parse_args()

    key_params = {'MODE_SCREEN': args.MODE_SCREEN, 'PLOT_BK': args.PLOT_BK,
                  'model_name': args.model_name, 'trackname': args.trackname, 'v_id': args.v_id}

    main(key_params)
