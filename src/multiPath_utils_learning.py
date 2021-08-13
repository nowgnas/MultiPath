# UTILITY-FUNCTIONS (LEARNING, TENSORFLOW)

import numpy as np
import tensorflow as tf  # tensorflow-2.0
from tensorflow import keras
import tensorflow_probability as tfp  # tensorflow-probability-0.8

ds = tfp.distributions


def get_log_softmax(tensor_in, axis=-1, min_value=1e-8, epsilon=1e-8):
    """ Gets log-softmax. """
    tensor_max = keras.backend.max(tensor_in, axis=axis, keepdims=True)
    tensor_in_ = tensor_in - tensor_max
    softmax = keras.backend.softmax(tensor_in_, axis=axis)
    softmax = keras.backend.clip(softmax, min_value=min_value, max_value=1.0)
    log_softmax = keras.backend.log(softmax + epsilon)
    return log_softmax


def get_log_sum_exp(tensor_in, axis=-1):
    """ Gets log-sum_exp. """
    tensor_max = keras.backend.max(tensor_in, axis=axis, keepdims=False)
    tensor_max_ = keras.backend.max(tensor_in, axis=axis, keepdims=True)
    tensor_in_ = tensor_in - tensor_max_
    tensor_exp = keras.backend.exp(tensor_in_)
    tensor_sum_exp = keras.backend.sum(tensor_exp, axis=axis, keepdims=False)
    tensor_log_sum_exp_ = keras.backend.log(tensor_sum_exp + 1e-8)
    tensor_log_sum_exp = tensor_log_sum_exp_ + tensor_max
    return tensor_log_sum_exp


# GAUSSIAN MODEL ------------------------------------------------------------------------------------------------------#
def get_kl_two_univariate_gaussians(mu1, sigma1, mu2, sigma2):
    """ KL divergence between two univariate: KL(p1|p2). """
    mean_diff = mu1 - mu2  # [..., dim]
    tmp0 = keras.backend.square(sigma1) + keras.backend.pow(mean_diff, 2)
    tmp1 = 2.0 * keras.backend.square(sigma2)
    tmp3 = tmp0 / tmp1
    tmp4 = sigma2 / (sigma1 + 1e-8)
    log_exponent = keras.backend.log(tmp4) + tmp3 - 0.5
    kl_div = keras.backend.sum(log_exponent, axis=1)
    return kl_div


def get_gdiag(mu, log_sigma):
    """ Gets Gaussian model (diag). """
    sigma = tf.math.exp(log_sigma)
    return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)


# GAUSSIAN MIXTURE MODEL (DIAG) ---------------------------------------------------------------------------------------#
def get_gmmdiag_components(dim_x, n_component_gmm, tensor_mu, tensor_log_sigma, tensor_log_pi, log_sigma_min=-10,
                           log_sigma_max=10):
    """ Gets components of Gaussian mixture model (diag).
    Args:
        dim_x: dimension of data
        n_component_gmm: number of gmm components
        tensor_mu: input tensor [batch_size, dim_x * n_component_gmm]
        tensor_log_sigma: input tensor [batch_size, dim_x * n_component_gmm]
        tensor_log_pi: input tensor [batch_size, n_component_gmm]
        log_sigma_min: minimum log-sigma
        log_sigma_max: maximum log-sigma
    Returns:
        mu_gmm: mean [batch_size, dim_x, n_component]
        log_sigma_gmm: log of standard-derivation [batch_size, dim_x, n_component]
        log_pi_gmm: log of fraction [batch_size, n_component]
    """
    # Reshape
    tensor_mu = keras.backend.reshape(tensor_mu, [-1, dim_x, n_component_gmm])  # [..., dim, num_of_components]
    tensor_log_sigma = keras.backend.reshape(tensor_log_sigma,
                                             [-1, dim_x, n_component_gmm])  # [..., dim, num_of_components]
    tensor_log_pi = keras.backend.reshape(tensor_log_pi, [-1, n_component_gmm])  # [..., num_of_components]

    mu_gmm, _log_sigma_gmm, _log_pi_gmm = tensor_mu, tensor_log_sigma, tensor_log_pi

    # log_sigma_gmm = keras.backend.clip(_log_sigma_gmm, min_value=log_sigma_min, max_value=log_sigma_max)
    log_sigma_min = tf.dtypes.cast(log_sigma_min, dtype=tf.float32)
    log_sigma_max = tf.dtypes.cast(log_sigma_max, dtype=tf.float32)
    log_sigma_gmm = log_sigma_min + (log_sigma_max - log_sigma_min) * keras.backend.sigmoid(_log_sigma_gmm)

    # log_pi_gmm = keras.backend.clip(log_pi_gmm, min_value=-10, max_value=10)
    log_pi_gmm = get_log_softmax(_log_pi_gmm, axis=1, min_value=1e-6)
    return mu_gmm, log_sigma_gmm, log_pi_gmm


def get_gmmdiag(mu, log_sigma, log_pi):
    """ Gets Gaussian mixture model (diag). """
    sigma_gmm = tf.math.exp(log_sigma)
    frac_gmm = tf.math.exp(log_pi)
    mu_gmm_list = tf.unstack(mu, axis=2)
    sigma_gmm_list = tf.unstack(sigma_gmm, axis=2)

    cat = ds.Categorical(probs=frac_gmm)
    cs = [ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma) \
          for mu, sigma in zip(mu_gmm_list, sigma_gmm_list)]

    gmm_diag = ds.Mixture(cat=cat, components=cs)

    return gmm_diag


def get_negloglikelihood_gmmdiag(gmm_diag, x, log_pi):
    """ Gets negative log likelihood from Gaussian mixture model (diag). """
    # TODO reshape x (64 36 1)
    log_likelihood = gmm_diag.log_prob(x)
    # TODO reshape x (64,)
    log_pi_prob = log_pi.log_prob(x)
    print(f'log prob {log_pi_prob}  log pi {log_pi}')

    # TODO sum normal D log prob, pi log prob
    neglog_likelihood = tf.negative(log_likelihood)
    loss_nll = tf.math.reduce_mean(neglog_likelihood, axis=-1)

    return loss_nll


def sample_gmmdiag(gmm_diag, num_sample=1):
    """ Samples from Gaussian mixture model (diag). """
    _sample_out = gmm_diag.sample(num_sample)
    sample_out = _sample_out.numpy()

    return sample_out


def train_sample_gmdiag(input_data, num_sample=1):
    sample_out = input_data.sample(num_sample)
    return sample_out


def train_log_prob(input_data):
    return input_data.log_prob(name='log_prob')
