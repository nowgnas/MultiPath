# PREDICTION WITH MIXTURE DENSITY NETWORK (MDN)
#
#   - INPUT: current-feature, previous trajectory
#   - TARGET: posterior trajectory
#
#   - Requirement:
#       tensorflow-2.0, tensorflow-probability-0.8

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

sys.path.insert(0, "../")

from src.multiPath_utils_learning import *
import tensorflow_probability as tfp  # tensorflow-probability-0.8


def get_default_hparams():
    """ Return default hyper-parameters """
    params_dict = {
        # Experiment Params:
        'n_component_gmm': 12,  # Number of mixture components
        'epochs': 50,  # How many times to go over the full train set (on average, since batches are drawn randomly).
        'batch_size': 64,  # Minibatch size (64, 128, 256).
        # Loss Params:
        'optimizer': 'adam',  # adam or sgd.
        'learning_rate': 0.001,  # Learning rate.
        'decay_rate': 0.98,  # Learning decay rate (exponential-decay).
        'grad_clip': 1.0,  # Gradient clipping. Recommend leaving at 1.0.
        # Data Params:
        'dim_p': 2,  # Dimension of data (delta-x, delta-y).
        'len_x': 6,  # Sequence length (previous-trajectory).
        'len_y': 18,  # Sequence length (posterior-trajectory).
        'sp_x': 0,  # Use sparse x (previous-trajectory).
        'sp_y': 0,  # Use sparse y (posterior-trajectory).
        'num_train': 89600,  # Number of train-data (262144).
        'num_test': 256,  # Number of test-data.
    }
    return params_dict


class Mdn(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Mdn, self).__init__()
        self.__dict__.update(kwargs)

        # Key hyper parameters
        self.use_image = int(self.hps.use_image)
        self.h_x = int(self.hps.len_x / 2) if self.hps.sp_x else int(self.hps.len_x)
        self.h_y = int(self.hps.len_y / 2) if self.hps.sp_y else int(self.hps.len_y)
        self.dim_p, self.dim_f = int(self.hps.dim_p), int(self.hps.dim_f)
        self.dim_y = self.h_y * self.hps.dim_p
        if self.use_image == 1:
            self.dim_i = (self.hps.dim_i[0], self.hps.dim_i[1], self.hps.dim_i[2])

        self.n_component_gmm = int(self.hps.n_component_gmm)

        # Build model
        self.nn = None
        self.build_model()  # Build model

        # Optimizer
        self.lr_decayed = keras.optimizers.schedules.ExponentialDecay(self.hps.learning_rate,
                                                                      decay_steps=self.hps.n_batch,
                                                                      decay_rate=self.hps.decay_rate, staircase=True)

        if self.hps.optimizer == 'adam':
            # self.optimizer = keras.optimizers.Adam(lr=self.hps.learning_rate, clipvalue=self.hps.grad_clip)
            self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_decayed, clipvalue=self.hps.grad_clip)
        elif self.hps.optimizer == 'sgd':
            self.optimizer = keras.optimizers.SGD(lr=self.lr_decayed, momentum=0.9, clipvalue=self.hps.grad_clip)
        else:
            raise ValueError("Unsupported Optimizer!")

    # MODEL DEFINITION ------------------------------------------------------------------------------------------------#
    def build_model(self):
        """ Builds model. """

        dim_out = (2 * self.dim_y + 1) * self.n_component_gmm

        if self.use_image == 1:
            _x_in = keras.Input(shape=(self.h_x, self.dim_p), dtype=tf.float32)
            _x_in_f = keras.layers.Flatten()(_x_in)
            _x1 = keras.layers.Dense(units=256, activation='relu')(_x_in_f)
            _xo = keras.layers.Dense(units=64, activation='relu')(_x1)

            _i_in = keras.Input(shape=self.dim_i, dtype=tf.float32)
            _i1 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation="relu")(_i_in)
            _i2 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation="relu")(_i1)
            _i3 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation="relu")(_i2)
            _i3f = keras.layers.Flatten()(_i3)
            _io = keras.layers.Dense(units=128, activation='relu')(_i3f)

            _c = keras.layers.concatenate([_xo, _io], axis=-1)
            _c1 = keras.layers.Dense(units=256, activation='relu')(_c)
            _c2 = keras.layers.Dense(units=256, activation='relu')(_c1)
            outputs_nn = keras.layers.Dense(units=dim_out)(_c2)
            inputs_nn = [_x_in, _i_in]

        else:
            _x_in = keras.Input(shape=(self.h_x, self.dim_p), dtype=tf.float32)
            _x_in_f = keras.layers.Flatten()(_x_in)
            _x1 = keras.layers.Dense(units=256, activation='relu')(_x_in_f)
            _xo = keras.layers.Dense(units=128, activation='relu')(_x1)

            _f_in = keras.Input(shape=(self.dim_f,), dtype=tf.float32)
            _f1 = keras.layers.Dense(units=256, activation='relu')(_f_in)
            _fo = keras.layers.Dense(units=128, activation='relu')(_f1)

            _c = keras.layers.concatenate([_xo, _fo], axis=-1)
            _c1 = keras.layers.Dense(units=256, activation='relu')(_c)
            _c2 = keras.layers.Dense(units=256, activation='relu')(_c1)
            outputs_nn = keras.layers.Dense(units=dim_out)(_c2)
            inputs_nn = [_x_in, _f_in]

        self.nn = keras.Model(inputs_nn, outputs_nn)
        print("Build model")

    def get_mdn_gmmdiag(self, x_data, f_data, i_data, y_data):
        """ Gets Gaussian mixture model (diag). """

        if self.use_image == 1:
            nn_out = self.nn([x_data, i_data])
        else:
            nn_out = self.nn([x_data, f_data])

        # for i in y_data:
        #     print(f'y data:{tf.print(i, output_stream=sys.stdout)}')
        # Set Gaussian mixture model
        indexes_split = [self.dim_y * self.n_component_gmm, self.dim_y * self.n_component_gmm, self.n_component_gmm]
        k = 3
        mu_gmm, log_sigma_gmm_tmp, log_frac_gmm = tf.split(nn_out, indexes_split, axis=1)
        data_y = tf.reshape(y_data, (-1, 36))
        data_mu = tf.reshape(mu_gmm, (-1, 12, 36))
        data_sigma = tf.reshape(log_sigma_gmm_tmp, (-1, 12, 36))
        sum_y_mu = data_y + data_mu[:, k, :]  # a_k (??) + mu
        # sigma
        sigma_sel = data_sigma[:, k, :]

        # pi
        pi_sel = log_frac_gmm[:, k]

        # TODO: Normal Distribution(a_k

        # cs = [ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma) for mu, sigma in zip(sum_y_mu, sigma_sel)]
        mu_gmm, log_sigma_gmm, log_pi_gmm = get_gmmdiag_components(self.dim_y, 1, sum_y_mu,
                                                                   sigma_sel, pi_sel,
                                                                   log_sigma_min=-10.0, log_sigma_max=10.0)

        gmm_diag = get_gmmdiag(mu_gmm, log_sigma_gmm, log_pi_gmm)

        return gmm_diag, mu_gmm, log_sigma_gmm, log_pi_gmm

    def sample(self, x_data, f_data, i_data, y_data, num_sample=1):
        """ Samples from GMM. """

        gmm_diag, mu_gmm, log_sigma_gmm, log_pi_gmm = self.get_mdn_gmmdiag(x_data, f_data, i_data, y_data)
        y_sample = sample_gmmdiag(gmm_diag, num_sample=num_sample)
        return y_sample

    # LOSS DEFINITION -------------------------------------------------------------------------------------------------#
    @tf.function
    def compute_loss(self, x_data, f_data, i_data, y_data):
        """" Computes loss for neg-log-likelihood. """
        y_data_r = keras.backend.reshape(y_data, shape=(-1, self.h_y * self.dim_p))
        gmm_diag, mu_gmm, log_sigma_gmm, log_pi_gmm = self.get_mdn_gmmdiag(x_data, f_data, i_data, y_data)
        loss_nll = get_negloglikelihood_gmmdiag(gmm_diag, y_data_r)
        return loss_nll

    # TRAIN -----------------------------------------------------------------------------------------------------------#
    def compute_gradients(self, x_data, f_data, i_data, y_data):
        """ Computes gradients for training. """
        with tf.GradientTape() as tape:
            loss_out = self.compute_loss(x_data, f_data, i_data, y_data)
        cg_out = tape.gradient(loss_out, self.nn.trainable_variables)
        return cg_out, loss_out

    @tf.function
    def train(self, x_data, f_data, i_data, y_data):
        """ Trains model. """
        gradients, loss_out = self.compute_gradients(x_data, f_data, i_data, y_data)
        self.optimizer.apply_gradients(zip(gradients, self.nn.trainable_variables))
        return loss_out

    # SAVE & LOAD -----------------------------------------------------------------------------------------------------#
    def save_trained_weights(self, filename):
        """ Saves weights of a trained model. 'weights' is path to h5 model\\weights file. """
        self.save_weights(filename)

    def load_trained_weights(self, filename):
        """ Loads weights of a pre-trained model. 'weights' is path to h5 model\\weights file. """
        self.load_weights(filename).expect_partial()
        print("Weights from {} loaded successfully".format(filename))

# EX: multipath pic 364 lane change