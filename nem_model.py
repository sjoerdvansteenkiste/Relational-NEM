#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell
from network import net, build_network
from sacred import Ingredient

nem = Ingredient('nem', ingredients=[net])


@nem.config
def cfg():
    # loss
    loss_inter_weight = 1.0     # weight for the inter-cluster loss
    loss_step_weights = 'all'   # all, last, or list of weights
    pixel_prior = {
        'p': 0.0,               # probability of success for pixel prior Bernoulli
    }

    # em
    k = 5                       # number of components
    nr_steps = 30               # number of EM steps
    pred_init = 0.0             # initial prediction used to compute the input


class NEMCell(RNNCell):
    """A RNNCell like implementation of N-EM."""
    @nem.capture
    def __init__(self, cell, input_shape, distribution, pred_init):
        self.cell = cell
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)
        self.input_size = input_shape
        self.gamma_shape = tf.TensorShape(input_shape.as_list()[:-1] + [1])
        self.distribution = distribution
        self.pred_init = pred_init

    @property
    def state_size(self):
        return self.cell.state_size, self.input_size, self.gamma_shape

    @property
    def output_size(self):
        return self.cell.output_size, self.input_size, self.gamma_shape

    def init_state(self, batch_size, K, dtype, gamma_init='gaussian'):
        # inner RNN hidden state init
        with tf.name_scope('inner_RNN_init'):
            h = self.cell.zero_state(batch_size * K, dtype)

        # initial prediction (B, K, W, H, C)
        with tf.name_scope('pred_init'):
            pred_shape = tf.stack([batch_size, K] + self.input_size.as_list())
            pred = tf.ones(shape=pred_shape, dtype=dtype) * self.pred_init

        # initial gamma (B, K, W, H, 1)
        with tf.name_scope('gamma_init_{}'.format(gamma_init)):
            gamma_shape = self.gamma_shape.as_list()
            shape = tf.stack([batch_size, K] + gamma_shape)

            # init with Gaussian distribution
            gamma = tf.abs(tf.random_normal(shape, dtype=dtype))
            gamma /= tf.reduce_sum(gamma, 1, keep_dims=True)

            # init with all 1 if K = 1
            if K == 1:
                gamma = tf.ones_like(gamma)

            return h, pred, gamma

    @staticmethod
    def delta_predictions(predictions, data):
        """Compute the derivative of the prediction wrt. to the loss.
        For binary and real with just μ this reduces to (predictions - data).
        :param predictions: (B, K, W, H, C)
           Note: This is a list to later support getting both μ and σ.
        :param data: (B, 1, W, H, C)

        :return: deltas (B, K, W, H, C)
        """
        with tf.name_scope('delta_predictions'):
            return data - predictions  # implicitly broadcasts over K

    @staticmethod
    @nem.capture
    def mask_rnn_inputs(rnn_inputs, gamma):
        """Mask the deltas (inputs to RNN) by gamma.
        :param rnn_inputs: (B, K, W, H, C)
            Note: This is a list to later support multiple inputs
        :param gamma: (B, K, W, H, 1)

        :return: masked deltas (B, K, W, H, C)
        """
        with tf.name_scope('mask_rnn_inputs'):
            gamma = tf.stop_gradient(gamma)

            return rnn_inputs * gamma  # implicitly broadcasts over C

    def run_inner_rnn(self, masked_deltas, h_old):
        with tf.name_scope('reshape_masked_deltas'):
            shape = tf.shape(masked_deltas)
            # print(masked_deltas.get_shape())
            batch_size = shape[0]
            K = shape[1]
            M = np.prod(self.input_size.as_list())
            reshaped_masked_deltas = tf.reshape(masked_deltas, tf.stack([batch_size * K, M]))

        preds, h_new = self.cell(reshaped_masked_deltas, h_old)

        return tf.reshape(preds, shape=shape), h_new

    def compute_em_probabilities(self, predictions, data, epsilon=1e-6):
        """Compute pixelwise loss of predictions (wrt. the data).

        :param predictions: (B, K, W, H, C)
        :param data: (B, 1, W, H, C)
        :return: local loss (B, K, W, H, 1)
        """

        with tf.name_scope('em_loss_{}'.format(self.distribution)):
            if self.distribution == 'bernoulli':
                mu = predictions
                loss = data * mu + (1 - data) * (1 - mu)
            else:
                raise ValueError('Unknown distribution_type: "{}"'.format(self.distribution))

            # sum loss over channels
            loss = tf.reduce_sum(loss, 4, keep_dims=True, name='reduce_channels')

            if epsilon > 0:
                # add epsilon to loss in order to prevent 0 gamma
                loss += epsilon

            return loss

    def e_step(self, preds, targets):
        with tf.name_scope('e_step'):
            probs = self.compute_em_probabilities(preds, targets)

            # compute the new gamma (E-step)
            gamma = probs / tf.reduce_sum(probs, 1, keep_dims=True)

            return gamma

    def __call__(self, inputs, state, scope=None):
        # unpack
        input_data, target_data = inputs
        h_old, preds_old, gamma_old = state

        # compute differences between prediction and input
        deltas = self.delta_predictions(preds_old, input_data)

        # mask with gamma
        masked_deltas = self.mask_rnn_inputs(deltas, gamma_old)

        # compute new predictions
        preds, h_new = self.run_inner_rnn(masked_deltas, h_old)

        # compute the new gammas
        gamma = self.e_step(preds, target_data)

        # pack and return
        outputs = (h_new, preds, gamma)

        return outputs, outputs


@nem.capture
def compute_prior(distribution, pixel_prior):
    """ Compute the prior over the input data.

    :return: prior (1, 1, 1, 1, 1)
    """

    if distribution == 'bernoulli':
        return tf.constant(pixel_prior['p'], shape=(1, 1, 1, 1, 1), name='prior')
    else:
        raise KeyError('Unknown distribution: "{}"'.format(distribution))


# log bci
def binomial_cross_entropy_loss(y, t):
    with tf.name_scope('binomial_ce'):
        clipped_y = tf.clip_by_value(y, 1e-6, 1. - 1.e-6)
        return -(t * tf.log(clipped_y) + (1. - t) * tf.log(1. - clipped_y))


# compute KL(p1, p2)
def kl_loss_bernoulli(p1, p2):
    with tf.name_scope('KL_loss'):
        return p1 * tf.log(tf.clip_by_value(p1 / tf.clip_by_value(p2, 1e-6, 1e6), 1e-6, 1e6)) + (1 - p1) * tf.log(tf.clip_by_value((1-p1)/tf.clip_by_value(1-p2, 1e-6, 1e6), 1e-6, 1e6))


@nem.capture
def compute_outer_loss(mu, gamma, target, prior, pixel_distribution, collision, loss_inter_weight):
    with tf.name_scope('outer_loss'):
        if pixel_distribution == 'bernoulli':
            intra_loss = binomial_cross_entropy_loss(mu, target)
            inter_loss = kl_loss_bernoulli(prior, mu)
        else:
            raise KeyError('Unknown pixel_distribution: "{}"'.format(pixel_distribution))

        # weigh losses by gamma and reduce by taking mean across B and sum across H, W, C, K
        # implemented as sum over all then divide by B
        batch_size = tf.to_float(tf.shape(target)[0])

        # compute rel losses
        r_intra_loss = tf.reduce_sum(collision * intra_loss * tf.stop_gradient(gamma)) / batch_size
        r_inter_loss = tf.reduce_sum(collision * inter_loss * (1. - tf.stop_gradient(gamma))) / batch_size

        # compute normal losses
        intra_loss = tf.reduce_sum(intra_loss * tf.stop_gradient(gamma)) / batch_size
        inter_loss = tf.reduce_sum(inter_loss * (1. - tf.stop_gradient(gamma))) / batch_size

        total_loss = intra_loss + loss_inter_weight * inter_loss
        r_total_loss = r_intra_loss + loss_inter_weight * r_inter_loss

        return total_loss, intra_loss, inter_loss, r_total_loss, r_intra_loss, r_inter_loss


@nem.capture
def compute_outer_ub_loss(pred, target, prior, pixel_distribution, collision, loss_inter_weight):
    with tf.name_scope('outer_loss'):
        max_pred = tf.reduce_max(pred, axis=1, keep_dims=True)
        if pixel_distribution == 'bernoulli':
            intra_ub_loss = binomial_cross_entropy_loss(max_pred, target)
            inter_ub_loss = kl_loss_bernoulli(prior, max_pred)
        else:
            raise KeyError('Unknown pixel_distribution: "{}"'.format(pixel_distribution))

        # weigh losses by gamma and reduce by taking mean across B and sum across H, W, C, K
        # implemented as sum over all then divide by B
        batch_size = tf.to_float(tf.shape(target)[0])

        # compute rel losses
        r_intra_ub_loss = tf.reduce_sum(collision * intra_ub_loss) / batch_size
        r_inter_ub_loss = tf.reduce_sum(collision * inter_ub_loss) / batch_size

        # compute normal losses
        intra_ub_loss = tf.reduce_sum(intra_ub_loss) / batch_size
        inter_ub_loss = tf.reduce_sum(inter_ub_loss) / batch_size

        total_ub_loss = intra_ub_loss + loss_inter_weight * inter_ub_loss
        r_total_ub_loss = r_intra_ub_loss + loss_inter_weight * r_inter_ub_loss

        return total_ub_loss, intra_ub_loss, inter_ub_loss, r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss


@nem.capture
def get_loss_step_weights(nr_steps, loss_step_weights):
    if loss_step_weights == 'all':
        return [1.0] * nr_steps
    elif loss_step_weights == 'last':
        loss_iter_weights = [0.0] * nr_steps
        loss_iter_weights[-1] = 1.0
        return loss_iter_weights
    elif isinstance(loss_step_weights, (list, tuple)):
        assert len(loss_step_weights) == nr_steps, len(loss_step_weights)
        return loss_step_weights
    else:
        raise KeyError('Unknown loss_step_weight type: "{}"'.format(loss_step_weights))


@nem.capture
def static_nem_iterations(input_data, target_data, k, collisions=None, actions=None):

    # Get dimensions
    input_shape = tf.shape(input_data)
    assert input_shape.get_shape()[0].value == 6, "Requires 6D input (T, B, K, W, H, C) but {}".format(input_shape.get_shape()[0].value)
    W, H, C = (x.value for x in input_data.get_shape()[-3:])

    # set pixel distribution
    pixel_dist = 'bernoulli'

    # set up inner cells and nem cells
    inner_cell = build_network(K=k)
    nem_cell = NEMCell(inner_cell, input_shape=(W, H, C), distribution=pixel_dist)

    # compute prior
    prior = compute_prior(distribution=pixel_dist)

    # get state initializer
    with tf.name_scope('initial_state'):
        hidden_state = nem_cell.init_state(input_shape[1], k, dtype=tf.float32)

    # build static iterations
    outputs = [hidden_state]
    total_losses, total_ub_losses, r_total_losses, r_total_ub_losses, other_losses, other_ub_losses, r_other_losses, r_other_ub_losses = [], [], [], [], [], [], [], []
    loss_step_weights = get_loss_step_weights()

    with tf.variable_scope('R-RNNEM') as varscope:
        for t, loss_weight in enumerate(loss_step_weights):
            varscope.reuse_variables() if t > 0 else None
            with tf.name_scope('step_{}'.format(t)):
                # compute inputs
                inputs = (input_data[t], target_data[t+1])

                # feed action through hidden state
                if actions is not None:
                    h_old, preds_old, gamma_old = hidden_state
                    h_old = {'state': h_old, 'action': actions[t]}
                    hidden_state = (h_old, preds_old, gamma_old)

                # run hidden cell
                hidden_state, output = nem_cell(inputs, hidden_state)
                theta, pred, gamma = output

                # set collision
                collision = tf.zeros((1, 1, 1, 1, 1)) if collisions is None else collisions[t]

                # compute nem losses
                total_loss, intra_loss, inter_loss, r_total_loss, r_intra_loss, r_inter_loss = compute_outer_loss(
                    pred, gamma, target_data[t+1], prior, pixel_distribution=pixel_dist, collision=collision)

                # compute estimated loss upper bound (which doesn't use E-step)
                total_ub_loss, intra_ub_loss, inter_ub_loss, r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss = \
                    compute_outer_ub_loss(pred, target_data[t+1], prior, pixel_distribution=pixel_dist, collision=collision)

            total_losses.append(loss_weight * total_loss)
            total_ub_losses.append(loss_weight * total_ub_loss)

            r_total_losses.append(loss_weight * r_total_loss)
            r_total_ub_losses.append(loss_weight * r_total_ub_loss)

            other_losses.append(tf.stack([total_loss, intra_loss, inter_loss]))
            other_ub_losses.append(tf.stack([total_ub_loss, intra_ub_loss, inter_ub_loss]))

            r_other_losses.append(tf.stack([r_total_loss, r_intra_loss, r_inter_loss]))
            r_other_ub_losses.append(tf.stack([r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss]))

            outputs.append(output)

    # collect outputs
    with tf.name_scope('collect_outputs'):
        thetas, preds, gammas = zip(*outputs)
        thetas = tf.stack(thetas)               # (T, 1, B*K, M)
        preds = tf.stack(preds)                 # (T, B, K, W, H, C)
        gammas = tf.stack(gammas)               # (T, B, K, W, H, C)
        other_losses = tf.stack(other_losses)   # (T, 3)
        other_ub_losses = tf.stack(other_ub_losses)   # (T, 3)
        r_other_losses = tf.stack(r_other_losses)
        r_other_ub_losses = tf.stack(r_other_ub_losses)
    with tf.name_scope('total_loss'):
        total_loss = tf.reduce_sum(tf.stack(total_losses)) / np.sum(loss_step_weights)
    with tf.name_scope('total_ub_loss'):
        total_ub_loss = tf.reduce_sum(tf.stack(total_ub_losses)) / np.sum(loss_step_weights)
    with tf.name_scope('r_total_loss'):
        r_total_loss = tf.reduce_sum(tf.stack(r_total_losses)) / np.sum(loss_step_weights)
    with tf.name_scope('r_total_ub_loss'):
        r_total_ub_loss = tf.reduce_sum(tf.stack(r_total_ub_losses)) / np.sum(loss_step_weights)

    return total_loss, total_ub_loss, r_total_loss, r_total_ub_loss, thetas, preds, gammas, other_losses, \
           other_ub_losses, r_other_losses, r_other_ub_losses


@nem.capture
def dynamic_nem_iteration(input_data, target_data, h_old, preds_old, gamma_old, k, collisions=None, actions=None):
    # Get dimensions
    input_shape = tf.shape(input_data)
    assert input_shape.get_shape()[0].value == 5, "Requires 5D input (B, K, W, H, C) but {}".format(input_shape.get_shape()[0].value)
    W, H, C = (x.value for x in input_data.get_shape()[-3:])

    # set pixel distribution
    pixel_dist = 'bernoulli'

    # set up inner cells and nem cells
    inner_cell = build_network(K=k)
    nem_cell = NEMCell(inner_cell, input_shape=(W, H, C), distribution=pixel_dist)

    # compute prior
    prior = compute_prior(distribution=pixel_dist)

    # build dynamic iteration
    with tf.variable_scope('R-RNNEM') as varscope:
        with tf.name_scope('step_{}'.format(0)):
            # compute inputs
            inputs = (input_data, target_data)

            # feed action through hidden state
            if actions is not None:
                h_old = {'state': h_old, 'action': actions}

            hidden_state = (h_old, preds_old, gamma_old)

            # run hidden cell
            hidden_state, output = nem_cell(inputs, hidden_state)

            theta, pred, gamma = output

            # set collision
            collision = tf.zeros((1, 1, 1, 1, 1)) if collisions is None else collisions

            # compute nem losses
            total_loss, intra_loss, inter_loss, r_total_loss, r_intra_loss, r_inter_loss = compute_outer_loss(
                pred, gamma, target_data, prior, pixel_distribution=pixel_dist, collision=collision)

            # compute estimated loss upper bound (which doesn't use E-step)
            total_ub_loss, intra_ub_loss, inter_ub_loss, r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss = \
                compute_outer_ub_loss(pred, target_data, prior, pixel_distribution=pixel_dist, collision=collision)

        other_losses = tf.stack([total_loss, intra_loss, inter_loss])
        other_ub_losses = tf.stack([total_ub_loss, intra_ub_loss, inter_ub_loss])

        r_other_losses = tf.stack([r_total_loss, r_intra_loss, r_inter_loss])
        r_other_ub_losses = tf.stack([r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss])

    return total_loss, total_ub_loss, r_total_loss, r_total_ub_loss, theta, pred, gamma, other_losses, \
           other_ub_losses, r_other_losses, r_other_ub_losses