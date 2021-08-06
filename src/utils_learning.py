# UTILITY-FUNCTIONS (LEARNING, TENSORFLOW)

import numpy as np
import tensorflow as tf  # tensorflow-2.0
from tensorflow import keras
import tensorflow_probability as tfp  # tensorflow-probability-0.8

ds = tfp.distributions


def normalize_input_helper(x, mu, sigma):
    """ Returns scaled tensor (batch_size, dim). """

    def subtract_mean_layer(params):
        """ Auxiliary function to feed into Lambda layer. Subtracts tensor. """
        _x, _mu = params
        _batch_size = keras.backend.shape(_x)[0]
        _mu_ext = keras.backend.tile(_mu, [_batch_size, 1])
        _tensor_out = (_x - _mu_ext)
        return _tensor_out

    def divide_sigma_layer(params):
        """ Auxiliary function to feed into Lambda layer. Divides tensor. """
        _x, _sigma = params
        _batch_size = keras.backend.shape(_x)[0]
        _sigma_ext = keras.backend.tile(_sigma, [_batch_size, 1])
        _tensor_out = _x / _sigma_ext
        return _tensor_out

    # We cannot simply use the operations and feed to the next layer, so a Lambda layer must be used
    _tensor_1 = keras.layers.Lambda(subtract_mean_layer)([x, mu])
    _tensor_o = keras.layers.Lambda(divide_sigma_layer)([_tensor_1, sigma])
    return _tensor_o


def sample_normal_dist_layer(mu, sigma):
    """ Returns a sample form normal distribution [batch_size, dim]. """

    def transform2layer(params):
        """ Auxiliary function to feed into Lambda layer:
         Gets a list of [mu, sigma] and returns a random tensor from the corresponding normal distribution. """
        _mu, _sigma = params
        colored_noise = _mu + _sigma * keras.backend.random_normal(shape=keras.backend.shape(_sigma), mean=0.0,
                                                                   stddev=1.0)
        return colored_noise

    # We cannot simply use the operations and feed to the next layer, so a Lambda layer must be used
    return keras.layers.Lambda(transform2layer)([mu, sigma])


def sample_gumbel_dist_layer(x, eps=1e-20):
    """ Samples from Gumbel(0, 1). """

    def uniform_dist_layer(params):
        """ Auxiliary function to feed into Lambda layer: Samples uniform distribution. """
        _x = params
        _u = keras.backend.random_uniform(keras.backend.shape(_x), minval=0, maxval=1)
        return _u

    def gumbel_dist_layer(params):
        """ Auxiliary function to feed into Lambda layer: Sample Gumbel distribution. """
        _u, _eps = params
        _g = -keras.backend.log(-keras.backend.log(_u + _eps) + _eps)
        return _g

    # We cannot simply use the operations and feed to the next layer, so a Lambda layer must be used
    _tensor_1 = keras.layers.Lambda(uniform_dist_layer)(x)
    _tensor_o = keras.layers.Lambda(gumbel_dist_layer)([_tensor_1, eps])
    return _tensor_o


def sample_gumbel_softmax_dist_layer(logits, temperature, hard=False):
    """ Samples from the Gumbel-Softmax distribution and optionally discretize.
      Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
      Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probability distribution that sums to 1 across classes
      """

    # Draw a sample from the Gumbel-Softmax distribution
    y_0 = logits + sample_gumbel_dist_layer(logits)
    layer_divide = keras.layers.Lambda(lambda x: x / temperature)
    y_1 = layer_divide(y_0)
    y = keras.activations.softmax(y_1)

    if hard:
        def transform2layer(params):
            """ Auxiliary function to feed into Lambda layer """
            _y = params
            _y_hard = keras.backend.cast(keras.backend.equal(_y, keras.backend.max(_y, 1, keep_dims=True)),
                                         keras.backend.dtype(_y))
            _y = keras.backend.stop_gradient(_y_hard - _y) + _y

            return _y

        # We cannot simply use the operations and feed to the next layer, so a Lambda layer must be used
        y = keras.layers.Lambda(transform2layer)(y)
    return y


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


def get_negloglikelihood_gmmdiag(gmm_diag, x):
    """ Gets negative log likelihood from Gaussian mixture model (diag). """
    log_likelihood = gmm_diag.log_prob(x)
    neglog_likelihood = tf.negative(log_likelihood)
    loss_nll = tf.math.reduce_mean(neglog_likelihood, axis=-1)

    return loss_nll


def sample_gmmdiag(gmm_diag, num_sample=1):
    """ Samples from Gaussian mixture model (diag). """
    _sample_out = gmm_diag.sample(num_sample)
    sample_out = _sample_out.numpy()

    return sample_out


# GAUSSIAN MIXTURE MODEL (2D) -----------------------------------------------------------------------------------------#
def get_gmm2d_components(n_component_gmm, tensor_mu, tensor_log_sigma, tensor_log_pi, tensor_rho, log_sigma_min=-10.0,
                         log_sigma_max=10.0):
    """ Gets components of Gaussian mixture model (2D).
        https://github.com/StanfordASL/TrafficWeavingCVAE/blob/master/utils/learning.py
        Sigma = [s1^2,   p*s1*s2      L = [s1,   0
                p*s1*s2, s2^2 ]           p*s2, sqrt(1-p^2)*s2]
        Sigma = L * L'
    Args:
        n_component_gmm: number of gmm components
        tensor_mu: input tensor [batch_size, 2 * n_component_gmm]
        tensor_log_sigma: input tensor [batch_size, 2 * n_component_gmm]
        tensor_log_pi: input tensor [batch_size, n_component_gmm]
        tensor_rho: input tensor [batch_size, n_component_gmm]
        log_sigma_min: minimum log-sigma
        log_sigma_max: maximum log-sigma
    Returns:
        mu_gmm: mean [batch_size, 2, n_component]
        log_sigma_gmm: log of standard-derivation [batch_size, 2, n_component]
        log_pi_gmm: log of fraction [batch_size, n_component]
        rho_gmm: correlation [batch_size, n_component]
        one_minus_rho_gmm: 1 - correlation [batch_size, n_component]
        l1_gmm: column 1 of L [batch_size, 2, n_component]
        l2_gmm: column 2 of L [batch_size, 2, n_component]
    """
    dim_x = 2

    # Reshape
    tensor_mu = keras.backend.reshape(tensor_mu, [-1, dim_x, n_component_gmm])  # [..., 2, num_of_components]
    tensor_log_sigma = keras.backend.reshape(tensor_log_sigma,
                                             [-1, dim_x, n_component_gmm])  # [..., 2, num_of_components]
    tensor_log_pi = keras.backend.reshape(tensor_log_pi, [-1, n_component_gmm])  # [..., num_of_components]
    tensor_rho = keras.backend.reshape(tensor_rho, [-1, n_component_gmm])  # [..., num_of_components]

    mu_gmm, _log_sigma_gmm, _log_pi_gmm, rho_gmm = tensor_mu, tensor_log_sigma, tensor_log_pi, tensor_rho

    # log_sigma_gmm = keras.backend.clip(_log_sigma_gmm, min_value=log_sigma_min, max_value=log_sigma_max)
    # sigma_gmm = keras.backend.exp(log_sigma_gmm)  # [..., 2, num_of_components]

    log_sigma_min = tf.dtypes.cast(log_sigma_min, dtype=tf.float32)
    log_sigma_max = tf.dtypes.cast(log_sigma_max, dtype=tf.float32)
    sigma_min, sigma_max = keras.backend.exp(log_sigma_min), keras.backend.exp(log_sigma_max)
    sigma_gmm = sigma_min + (sigma_max - sigma_min) * keras.backend.sigmoid(_log_sigma_gmm)
    log_sigma_gmm = keras.backend.log(sigma_gmm)

    # _log_pi_gmm = keras.backend.clip(_log_pi_gmm, min_value=-10, max_value=10)
    # _log_pi_gmm = 20 * keras.backend.tanh(_log_pi_gmm)
    log_pi_gmm = get_log_softmax(_log_pi_gmm, axis=1)

    rho_gmm = keras.backend.tanh(rho_gmm)

    one_minus_rho_gmm = 1 - keras.backend.square(rho_gmm)
    one_minus_rho_gmm = keras.backend.clip(one_minus_rho_gmm, min_value=1e-8, max_value=1.0)

    l1_gmm = sigma_gmm * keras.backend.stack([keras.backend.ones_like(rho_gmm), rho_gmm], axis=1)
    # [..., 2, num_of_components] (column 1 of L)
    l2_gmm = sigma_gmm * keras.backend.stack([keras.backend.zeros_like(rho_gmm),
                                              keras.backend.sqrt(one_minus_rho_gmm)], axis=1)
    # [..., 2, num_of_components] (column 2 of L)

    return mu_gmm, log_sigma_gmm, log_pi_gmm, rho_gmm, one_minus_rho_gmm, l1_gmm, l2_gmm


def get_negloglikelihood_gmm2d(n_component_gmm, mu_gmm, log_sigma_gmm, log_pi_gmm, rho_gmm, one_minus_rho_gmm, x):
    """ Get negative log likelihood from Gaussian mixture model (2d). """
    epsilon = 1e-8

    # x: [..., 2]
    log_sigma_sum = keras.backend.sum(log_sigma_gmm, axis=1)  # [..., num_of_components]
    sigma = keras.backend.exp(log_sigma_gmm)  # [..., 2, num_of_components]
    sigma_prod = keras.backend.prod(sigma, axis=1)  # [..., num_of_components]
    log_one_minus_rho_gmm = keras.backend.log(one_minus_rho_gmm)  # [..., num_of_components]
    sigma_c = sigma + epsilon

    x = keras.backend.expand_dims(x, axis=-1)  # [..., 2, 1]
    x_t = keras.backend.tile(x, [1, 1, n_component_gmm])  # [..., 2, num_of_components]
    dx = x_t - mu_gmm  # [..., 2, num_of_components]
    dx_scaled = dx / sigma_c  # [..., 2, num_of_components]
    dx_scaled_sq = keras.backend.square(dx_scaled)  # [..., 2, num_of_components]
    dx_prod = keras.backend.prod(dx, axis=1)  # [..., num_of_components]
    dx_prod_scaled = dx_prod / (sigma_prod + epsilon)  # [..., num_of_components]

    exponent_1 = keras.backend.sum(dx_scaled_sq, axis=1)  # [..., num_of_components]
    exponent_2 = -2.0 * rho_gmm * dx_prod_scaled  # [..., num_of_components]
    exponent_3 = exponent_1 + exponent_2  # [..., num_of_components]
    exponent = -0.5 * (exponent_3 / one_minus_rho_gmm)

    const_tmp = tf.constant(np.log(2.0 * np.pi), dtype=tf.float32)
    log_prob_i_ = const_tmp + log_sigma_sum + 0.5 * log_one_minus_rho_gmm  # [..., num_of_components]
    log_prob_i = -log_prob_i_ + exponent  # [..., num_of_components]

    log_prob = tf.math.reduce_logsumexp(log_pi_gmm + log_prob_i, axis=1)  # [...]
    # log_prob = get_log_sum_exp(log_pi_gmm + log_prob_i, axis=1)  # [...]

    loss_negloglikelihood = -1.0 * log_prob  # [...]

    return loss_negloglikelihood


def sample_gmm2d(n_component_gmm, mu_gmm, log_pi_gmm, rho_gmm, l1_gmm, l2_gmm):
    """ Samples from Gaussian mixture model (2d).
    mu_gmm: mean [batch_size, 2, n_component]
    log_pi_gmm: log of fraction [batch_size, n_component]
    rho_gmm: correlation [batch_size, n_component]
    l1_gmm [batch_size, 2, num_of_components] (column 1 of L)
    l2_gmm [batch_size, 2, num_of_components] (column 2 of L)

    Sigma = [s1^2,   p*s1*s2      L = [s1,   0
             p*s1*s2, s2^2 ]           p*s2, sqrt(1-p^2)*s2]

    Sigma = L * L'
    Sample = mu + L * z (z = independent random variables sampled from the standard normal distribution)
    """
    shape_tmp = keras.backend.shape(rho_gmm)
    rand_tmp1_ = keras.backend.expand_dims(keras.backend.random_normal(shape_tmp), axis=1)
    # [..., 1, num_of_components]
    rand_tmp2_ = keras.backend.expand_dims(keras.backend.random_normal(shape_tmp), axis=1)
    # [..., 1, num_of_components]
    rand_tmp1 = keras.backend.tile(rand_tmp1_, [1, 2, 1])  # [..., 2, num_of_components]
    rand_tmp2 = keras.backend.tile(rand_tmp2_, [1, 2, 1])  # [..., 2, num_of_components]
    mvn_sample = mu_gmm + (l1_gmm * rand_tmp1 + l2_gmm * rand_tmp2)
    # (manual 2x2 matmul)
    # [..., 2, num_of_components]

    cat = tfp.distributions.Categorical(logits=log_pi_gmm)
    cat_sample = cat.sample()  # [...]

    # cat_sample = tf.random.categorical(logits=log_pi_gmm, num_samples=1)  # [..., 1]
    # cat_sample = keras.backend.flatten(cat_sample)

    selector_ = keras.backend.expand_dims(keras.backend.one_hot(cat_sample, n_component_gmm), axis=1)
    # [..., 1, num_of_components]
    selector = keras.backend.tile(selector_, [1, 2, 1])  # [..., 2, num_of_components]

    sample_out = keras.backend.sum(mvn_sample * selector, axis=-1)  # [..., 2]
    return sample_out
