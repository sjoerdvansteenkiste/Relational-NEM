#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # no INFO/WARN logs from Tensorflow

import time
import utils
import threading
import numpy as np
import tensorflow as tf
from tensorflow.contrib import distributions as dist

from sacred import Experiment
from sacred.utils import get_by_dotted_path
from datasets import ds
from datasets import InputPipeLine
from nem_model import nem, static_nem_iterations, dynamic_nem_iteration, get_loss_step_weights
from network import net

ex = Experiment("R-NEM", ingredients=[ds, nem, net])


# noinspection PyUnusedLocal
@ex.config
def cfg():
    noise = {
        'noise_type': 'bitflip',                        # noise type
        'prob': 0.2,                                    # probability of annihilating the pixel
    }
    training = {
        'optimizer': 'adam',                            # {adam, sgd, momentum, adadelta, adagrad, rmsprop}
        'params': {
            'learning_rate': 0.001,                     # float
        },
        'max_patience': 10,                             # number of epochs to wait before early stopping
        'batch_size': 64,
        'max_epoch': 500,
        'clip_gradients': None,                         # maximum norm of gradients
        'debug_samples': [3, 37, 54],                   # sample ids to generate plots for (None, int, list)
        'save_epochs': [1, 5, 10, 20, 50, 100]          # at what epochs to save the model independent of valid loss
    }
    validation = {
        'batch_size': training['batch_size'],
        'debug_samples': [0, 1, 2]                      # sample ids to generate plots for (None, int, list)
    }

    feed_actions = False                                # whether to feed the actions (RL) via the recurrent state
    record_grouping_score = True                        # whether to use grouping to compute ARI/AMI scores
    record_relational_loss = 'collisions'               # use {events, collisions} to compute rel. losses or None

    dt = 10                                             # how many steps to include in the last loss
    log_dir = 'debug_out'                               # directory to dump logs and debug plots
    net_path = None                                     # path of to network file to initialize weights with

    # config to control run_from_file
    run_config = {
        'usage': 'test',                                # what dataset to use {training, validation, test}
        'batch_size': 100,
        'rollout_steps': 10,
        'debug_samples': [0, 1, 2],                      # sample ids to generate plots for (None, int, list)
    }


ex.add_named_config('no_score', {'record_grouping_score': False})
ex.add_named_config('no_collisions', {'record_relational_loss': None})


@ex.capture
def add_noise(data, noise):
    noise_type = noise['noise_type']
    if noise_type in ['None', 'none', None]:
        return data

    with tf.name_scope('input_noise'):
        shape = tf.stack([s.value if s.value is not None else tf.shape(data)[i]
                         for i, s in enumerate(data.get_shape())])

        if noise_type == 'bitflip':
            noise_dist = dist.Bernoulli(probs=noise['prob'], dtype=data.dtype)
            n = noise_dist.sample(shape)
            corrupted = data + n - 2 * data * n  # hacky way of implementing (data XOR n)
        else:
            raise KeyError('Unknown noise_type "{}"'.format(noise_type))

        corrupted.set_shape(data.get_shape())
        return corrupted


@ex.capture(prefix='training')
def set_up_optimizer(loss, optimizer, params, clip_gradients):
    opt = {
        'adam': tf.train.AdamOptimizer,
        'sgd': tf.train.GradientDescentOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'rmsprop': tf.train.RMSPropOptimizer
    }[optimizer](**params)
    grads_and_vars = opt.compute_gradients(loss)
    if clip_gradients is not None:
        grads_and_vars = [(tf.clip_by_norm(grad, clip_gradients), var)
                          for grad, var in grads_and_vars]

    return opt, opt.apply_gradients(grads_and_vars)


@ex.capture
def build_dynamic_graph(features, targets, gammas_old, thetas_old, preds_old, network, groups=None, collisions=None, actions=None):
    # Training graph
    features_corrupted = add_noise(features)
    loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses, r_other_losses, r_other_ub_losses = dynamic_nem_iteration(
        input_data=features_corrupted, target_data=targets, gamma_old=gammas_old, h_old=thetas_old, preds_old=preds_old, collisions=collisions, actions=actions)

    graph = {
        'inputs': features,
        'corrupted': features_corrupted,
        'targets': targets,
        'loss': loss,
        'ub_loss': ub_loss,
        'r_loss': r_loss,
        'r_ub_loss': r_ub_loss,
        'gammas_old': gammas_old,
        'thetas_old': thetas_old,
        'preds_old': preds_old,
        'gammas': gammas,
        'thetas': thetas,
        'preds': preds,
        'other_losses': other_losses,
        'other_ub_losses': other_ub_losses,
        'r_other_losses': r_other_losses,
        'r_other_ub_losses': r_other_ub_losses,
    }

    # compute grouping info
    if groups is not None:
        graph['groups'] = groups

    if collisions is not None:
        graph['collisions'] = collisions

    # add actions to the graph
    if actions is not None:
        graph['actions'] = actions

    # if NPE with a non-empty attention block.
    if network['recurrent'][0]['name'] == 'npe' and len(network['recurrent'][0]['attention']) > 0:
        k = gammas.shape[1].value

        ns = tf.contrib.framework.get_name_scope()
        g = tf.get_default_graph()
        attentions = [g.get_tensor_by_name("{}/R-RNNEM/step_0/NPE/Sigmoid:0".format(ns))]
        attentions = tf.stack(attentions, axis=0)
        attentions = tf.reshape(attentions, [1, -1, k, k - 1])
        graph['attentions'] = attentions  # in order

    return graph


@net.capture
def build_rollout_graph(inputs, batch_size, k, recurrent):
    feature_shape = [s.value for s in inputs['features'].shape[2:]]
    groups_shape = [s.value for s in inputs['groups'].shape[2:]] if inputs.get('groups', None) is not None else None
    actions_shape = [s.value for s in inputs['actions'].shape[2:]] if inputs.get('actions', None) is not None else None

    with tf.name_scope('rollout'):
        X_rollout_shape = [batch_size] + feature_shape
        X_rollout = tf.placeholder(tf.float32, shape=X_rollout_shape)

        Y_rollout = tf.placeholder(tf.float32, shape=X_rollout_shape)

        gamma_rollout_shape = [batch_size, k] + feature_shape[1:]
        gamma_rollout = tf.placeholder(tf.float32, shape=gamma_rollout_shape)

        theta_rollout_shape = [batch_size*k, recurrent[0]['size']]
        theta_rollout = tf.placeholder(tf.float32, shape=theta_rollout_shape)

        pred_rollout = tf.placeholder(tf.float32, shape=gamma_rollout_shape)

        if inputs.get('groups', None) is not None:
            G_rollout_shape = [batch_size] + groups_shape
            G_rollout = tf.placeholder(tf.float32, shape=G_rollout_shape)
        else:
            G_rollout = None

        if inputs.get('collisions', None) is not None:
            collisions_rollout_shape = [batch_size] + feature_shape
            collision_rollout = tf.placeholder(tf.float32, shape=collisions_rollout_shape)
        elif inputs.get('events', None) is not None:
            collisions_rollout_shape = [batch_size, 1, 1, 1, 1]
            collision_rollout = tf.placeholder(tf.float32, shape=collisions_rollout_shape)
        else:
            collision_rollout = None

        if inputs.get('actions', None) is not None:
            A_rollout_shape = [batch_size] + actions_shape
            A_rollout = tf.placeholder(tf.float32, shape=A_rollout_shape)
        else:
            A_rollout = None

        graph = build_dynamic_graph(X_rollout, Y_rollout, gamma_rollout, theta_rollout, pred_rollout,
                                              groups=G_rollout, collisions=collision_rollout, actions=A_rollout)
        return graph


@ex.capture
def build_graph(features, network, groups=None, collisions=None, actions=None):
    
    # Training graph
    features_corrupted = add_noise(features)
    loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses, r_other_losses, \
    r_other_ub_losses = static_nem_iterations(features_corrupted, features,
                                              collisions=collisions, actions=actions)

    graph = {
        'inputs': features,
        'corrupted': features_corrupted,
        'loss': loss,
        'ub_loss': ub_loss,
        'r_loss': r_loss,
        'r_ub_loss': r_ub_loss,
        'gammas': gammas,
        'thetas': thetas,
        'preds': preds,
        'other_losses': other_losses,
        'other_ub_losses': other_ub_losses,
        'r_other_losses': r_other_losses,
        'r_other_ub_losses': r_other_ub_losses,
    }

    # compute grouping info
    if groups is not None:
        graph['groups'] = groups
        graph['ARI'] = utils.tf_adjusted_rand_index(groups, gammas, get_loss_step_weights())

    # add actions to the graph
    if actions is not None:
        graph['actions'] = actions

    # if NPE with a non-empty attention block.
    if network['recurrent'][0]['name'] == 'npe' and len(network['recurrent'][0]['attention']) > 0:
        nr_iters = gammas.shape[0].value
        k = gammas.shape[2].value

        attentions = []
        ns = tf.contrib.framework.get_name_scope()
        g = tf.get_default_graph()
        for i in range(nr_iters-1):
            attention = g.get_tensor_by_name("{}/R-RNNEM/step_{}/NPE/Sigmoid:0".format(ns, i))
            attentions.append(attention)

        attentions = tf.stack(attentions, axis=0)
        attentions = tf.reshape(attentions, [nr_iters-1, -1, k, k-1])

        graph['attentions'] = attentions  # in order

    return graph


def build_debug_graph(inputs):
    nr_iters = inputs['features'].shape[0]
    feature_shape = [s.value for s in inputs['features'].shape[2:]]
    groups_shape = [s.value for s in inputs['groups'].shape[2:]] if inputs.get('groups', None) is not None else None
    actions_shape = [s.value for s in inputs['actions'].shape[2:]] if inputs.get('actions', None) is not None else None

    with tf.name_scope('debug'):
        X_debug_shape = [nr_iters, None] + feature_shape
        X_debug = tf.placeholder(tf.float32, shape=X_debug_shape)

        if inputs.get('groups', None) is not None:
            G_debug_shape = [nr_iters, None] + groups_shape
            G_debug = tf.placeholder(tf.float32, shape=G_debug_shape)
        else:
            G_debug = None

        if inputs.get('actions', None) is not None:
            A_debug_shape = [nr_iters, None] + actions_shape
            A_debug = tf.placeholder(tf.float32, shape=A_debug_shape)
        else:
            A_debug = None

        graph = build_graph(X_debug, groups=G_debug, actions=A_debug)
        return graph


@ex.capture
def build_graphs(train_inputs, valid_inputs, record_relational_loss):
    
    # Build Graph
    varscope = tf.get_variable_scope()
    with tf.name_scope("train"):
        train_graph = build_graph(train_inputs['features'],
                                  groups=train_inputs.get('groups', None),
                                  collisions=train_inputs.get(record_relational_loss, None),
                                  actions=train_inputs.get('actions', None)
                                  )
        opt, train_op = set_up_optimizer(train_graph['loss'])

    varscope.reuse_variables()
    with tf.name_scope("valid"):
        valid_graph = build_graph(valid_inputs['features'],
                                  groups=valid_inputs.get('groups', None),
                                  collisions=valid_inputs.get(record_relational_loss, None),
                                  actions=valid_inputs.get('actions', None))

    debug_graph = build_debug_graph(valid_inputs)

    return train_op, train_graph, valid_graph, debug_graph


@ex.capture
def create_curve_plots(name, plot_dict, coarse_range, fine_range, log_dir):
    import matplotlib.pyplot as plt
    fig = utils.curve_plot(plot_dict, coarse_range, fine_range)
    fig.suptitle(name)
    fig.savefig(os.path.join(log_dir, name + '_curve.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


@ex.capture
def create_debug_plots(name, debug_out, sample_indices, log_dir, debug_groups=None):
    import matplotlib.pyplot as plt

    if debug_groups is not None:
        scores, confidencess = utils.evaluate_groups_seq(debug_groups[1:], debug_out['gammas'][1:], get_loss_step_weights())
    else:
        scores, confidencess = len(sample_indices) * [0.0], len(sample_indices) * [0.0]

    # produce overview plot
    for i, nr in enumerate(sample_indices):
        fig = utils.overview_plot(i, **debug_out)
        fig.suptitle(name + ', sample {},  AMI Score: {:.3f} ({:.3f}) '.format(nr, scores[i], confidencess[i]))
        fig.savefig(os.path.join(log_dir, name + '_{}.png'.format(nr)), bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def populate_debug_out(session, debug_graph, pipe_line, debug_samples, name):
    idxs = debug_samples if isinstance(debug_samples, list) else [debug_samples]

    out_list = ['features']
    out_list.append('groups') if debug_graph.get('groups', None) is not None else None
    out_list.append('actions') if debug_graph.get('actions', None) is not None else None

    debug_data = pipe_line.get_debug_samples(idxs, out_list=out_list)

    feed_dict = {debug_graph['inputs']: debug_data['features']}

    if debug_data.get('groups', None) is not None:
        feed_dict[debug_graph['groups']] = debug_data['groups']

    if debug_data.get('actions', None) is not None:
        feed_dict[debug_graph['actions']] = debug_data['actions']

    debug_out = session.run(debug_graph, feed_dict=feed_dict)
    create_debug_plots(name, debug_out, idxs, debug_groups=debug_data.get('groups', None))


def run_epoch(session, pipe_line, graph, debug_graph, debug_samples, debug_name, train_op=None):
    fetches = [graph['loss'], graph['ub_loss'], graph['r_loss'], graph['r_ub_loss'], graph['other_losses'],
               graph['other_ub_losses'], graph['r_other_losses'], graph['r_other_ub_losses']]

    fetches.append(graph['ARI']) if graph.get('ARI', None) is not None else None
    fetches.append(train_op) if train_op is not None else None

    losses, ub_losses, r_losses, r_ub_losses, others, others_ub, r_others, r_others_ub, ari_scores = [], [], [], [], [], [], [], [], []
    # run through the epoch
    for b in range(pipe_line.get_n_batches()):
        # run batch
        out = session.run(fetches=fetches)

        # total losses (and upperbound)
        losses.append(out[0])
        ub_losses.append(out[1])

        # total relational losses (and upperbound)
        r_losses.append(out[2])
        r_ub_losses.append(out[3])

        # other losses (and upperbound)
        others.append(out[4])
        others_ub.append(out[5])

        # other relational losses (and upperbound)
        r_others.append(out[6])
        r_others_ub.append(out[7])

        # ARI
        ari_scores.append(out[8] if graph.get('ARI', None) is not None else (0., 0., 0., 0.))

    if debug_samples is not None:
        populate_debug_out(session, debug_graph, pipe_line, debug_samples, debug_name)

    # build log dict
    log_dict = {
        'loss': float(np.mean(losses)),
        'ub_loss': float(np.mean(ub_losses)),
        'r_loss': float(np.mean(r_losses)),
        'r_ub_loss': float(np.mean(r_ub_losses)),
        'others': np.mean(others, axis=0),
        'others_ub': np.mean(others_ub, axis=0),
        'r_others': np.mean(r_others, axis=0),
        'r_others_ub': np.mean(r_others_ub, axis=0),
        'score': np.mean(ari_scores, axis=0)[0],
        'score_last': np.mean(ari_scores, axis=0)[1],
        'score_conf': np.mean(ari_scores, axis=0)[2],
        'score_last_conf': np.mean(ari_scores, axis=0)[3]
    }

    return log_dict


@ex.capture
def add_log(key, value, _run):
    if 'logs' not in _run.info:
        _run.info['logs'] = {}
    logs = _run.info['logs']
    split_path = key.split('.')
    current = logs
    for p in split_path[:-1]:
        if p not in current:
            current[p] = {}
        current = current[p]

    final_key = split_path[-1]
    if final_key not in current:
        current[final_key] = []
    entries = current[final_key]
    entries.append(value)


@ex.capture
def get_logs(key, _run):
    logs = _run.info.get('logs', {})
    return get_by_dotted_path(logs, key)


def log_log_dict(usage, log_dict):
    for log_key, value in log_dict.items():
        add_log('{}.{}'.format(usage, log_key), value)


def print_log_dict(log_dict, usage, t, dt, s_loss_weights, dt_s_loss_weights):
    print("%s Loss: %.3f (UB: %.3f), Relational Loss: %.3f (UB: %.3f), Score: %.3f (conf: %0.3f), Last Score:"
          " %.3f (conf: %.3f) took %.3fs" % (usage, log_dict['loss'], log_dict['ub_loss'], log_dict['r_loss'],
                                             log_dict['r_ub_loss'], log_dict['score'], log_dict['score_conf'],
                                             log_dict['score_last'], log_dict['score_last_conf'], time.time() - t))

    print("    other losses: {}".format(", ".join(["%.2f (UB: %.2f)" %
          (log_dict['others'][:, i].sum(0) / s_loss_weights, log_dict['others_ub'][:, i].sum(0) / s_loss_weights)
           for i in range(len(log_dict['others'][0]))])))
    print("        last {} steps avg: {}".format(dt, ", ".join(["%.2f (UB: %.2f)" %
          (log_dict['others'][-dt:, i].sum(0) / dt_s_loss_weights,
           log_dict['others_ub'][-dt:, i].sum(0) / dt_s_loss_weights) for i in range(len(log_dict['others'][0]))])))

    print("    other relational losses: {}".format(", ".join(["%.2f (UB: %.2f)" %
          (log_dict['r_others'][:, i].sum(0) / s_loss_weights, log_dict['r_others_ub'][:, i].sum(0) / s_loss_weights)
           for i in range(len(log_dict['r_others'][0]))])))
    print("        last {} steps avg: {}".format(dt, ", ".join(["%.2f (UB: %.2f)" %
          (log_dict['r_others'][-dt:, i].sum(0) / dt_s_loss_weights,
           log_dict['r_others_ub'][-dt:, i].sum(0) / dt_s_loss_weights) for i in range(len(log_dict['r_others'][0]))])))


@ex.command
def rollout_from_file(record_grouping_score, record_relational_loss, feed_actions, run_config, nem, dt, log_dir, seed, net_path=None):
    tf.set_random_seed(seed)

    # load network weights (default is log_dir/best if net_path is not set)
    net_path = os.path.abspath(os.path.join(log_dir, 'best')) if net_path is None else net_path
    usage = run_config['usage']

    # prep weights for print out
    loss_step_weights = get_loss_step_weights()
    s_loss_weights = np.sum(loss_step_weights)
    dt_s_loss_weights = np.sum(loss_step_weights[-dt:])

    with tf.Graph().as_default() as g:
        # Set up Data
        batch_size = run_config['batch_size']
        nr_steps = nem['nr_steps'] + run_config['rollout_steps'] + 1
        out_list = ['features']
        out_list.append('groups') if record_grouping_score else None
        out_list.append(record_relational_loss) if record_relational_loss else None
        out_list.append('actions') if feed_actions else None

        inputs = InputPipeLine(usage, shuffle=False, sequence_length=nr_steps, out_list=out_list, batch_size=batch_size)

        # Build Graph
        graph = build_rollout_graph(inputs.output, batch_size, nem['k'])

        start_time = time.time()
        with tf.Session(graph=g) as session:
            saver = tf.train.Saver()
            saver.restore(session, net_path)

            # produce data
            fetches = [graph['loss'], graph['ub_loss'], graph['r_loss'], graph['r_ub_loss'], graph['other_losses'],
                       graph['other_ub_losses'], graph['r_other_losses'], graph['r_other_ub_losses'],
                       graph['corrupted'], graph['gammas'], graph['thetas'], graph['preds']]

            # create loss dict
            loss_dict = {'loss': [], 'ub_loss': [], 'r_loss': [], 'r_ub_loss': [], 'others': [], 'others_ub': [],
                         'r_others': [], 'r_others_ub': []}

            # debug out
            for b in range(inputs.get_n_batches()):
                idxs = list(range(b*batch_size, (b+1) * batch_size))
                input_data = inputs.get_debug_samples(idxs, out_list=out_list)

                # create empty list
                loss_dict['loss'].append([])
                loss_dict['ub_loss'].append([])
                loss_dict['r_loss'].append([])
                loss_dict['r_ub_loss'].append([])
                loss_dict['others'].append([])
                loss_dict['others_ub'].append([])
                loss_dict['r_others'].append([])
                loss_dict['r_others_ub'].append([])

                # init
                with tf.name_scope('initial_state'):
                    # inner RNN hidden state init
                    with tf.name_scope('inner_RNN_init'):
                        theta = np.zeros((batch_size * nem['k'], 250), dtype=np.float32)

                    # initial prediction (B, K, W, H, C)
                    with tf.name_scope('pred_init'):
                        pred = np.ones((batch_size, nem['k'], 64, 64, 1), dtype=np.float32) * nem['pred_init']

                    # initial gamma (B, K, W, H, 1)
                    with tf.name_scope('gamma_init_gaussian'):
                        # init with Gaussian distribution
                        gamma = np.abs(np.random.randn(batch_size, nem['k'], 64, 64, 1))
                        gamma /= np.sum(gamma, axis=1, keepdims=True)

                        # init with all 1 if K = 1
                        if nem['k'] == 1:
                            gamma = np.ones_like(gamma)

                corrupted, scores, gammas, thetas, preds = [], [], [gamma], [theta], [pred]

                # run rollout steps
                for t in range(nem['nr_steps'] + run_config['rollout_steps']):

                    # build feed dict
                    feed_dict = {graph['targets']: input_data['features'][t + 1],
                                 graph['gammas_old']: gamma,
                                 graph['thetas_old']: theta,
                                 graph['preds_old']: pred}

                    # decided if rollout or real data
                    if t < nem['nr_steps']:
                        feed_dict[graph['inputs']] = input_data['features'][t]
                    else:
                        feed_dict[graph['inputs']] = np.sum(gamma * pred, 1, keepdims=True)

                    if input_data.get('groups', None) is not None:
                        feed_dict[graph['groups']] = input_data['groups'][t+1]

                    if input_data.get('collisions', None) is not None:
                        feed_dict[graph['collisions']] = input_data['collisions'][t]
                    elif input_data.get('events', None) is not None:
                        feed_dict[graph['collisions']] = input_data['events'][t]

                    if input_data.get('actions', None) is not None:
                        feed_dict[graph['actions']] = input_data['actions'][t]

                    # run forward pass
                    out = session.run(fetches, feed_dict=feed_dict)

                    # log results for iteration
                    corr, gamma, theta, pred = out[-4:]

                    # re-compute gamma if rollout
                    if t >= nem['nr_steps']:
                        truth = np.max(pred, axis=1, keepdims=True)

                        # avoid disappearing by scaling or sampling
                        truth[truth > 0.1] = 1.0
                        truth[truth <= 0.1] = 0.0

                        # compute probs
                        probs = truth * pred + (1 - truth) * (1 - pred)

                        # add epsilon to probs in order to prevent 0 gamma
                        probs += 1e-6

                        # compute the new gamma (E-step) or set to one for k=1
                        gamma = probs / np.sum(probs, 1, keepdims=True) if nem['k'] > 1 else np.ones_like(gamma)

                    corrupted.append(corr)
                    gammas.append(gamma)
                    thetas.append(theta)
                    preds.append(pred)

                    # log losses
                    loss_dict['loss'][-1].append(out[0])
                    loss_dict['ub_loss'][-1].append(out[1])
                    loss_dict['r_loss'][-1].append(out[2])
                    loss_dict['r_ub_loss'][-1].append(out[3])
                    loss_dict['others'][-1].append(out[4])
                    loss_dict['others_ub'][-1].append(out[5])
                    loss_dict['r_others'][-1].append(out[6])
                    loss_dict['r_others_ub'][-1].append(out[7])

                # build plot dict if needed
                out_dict = {
                    'inputs': input_data['features'],
                    'corrupted': np.array(corrupted),
                    'gammas': np.array(gammas),
                    'preds': np.array(preds),
                }

                # create debug plots for entries in first batch
                if b == 0 and run_config.get('debug_samples', None):
                    create_debug_plots('rollout_{}'.format(usage), out_dict, run_config['debug_samples'])

            # build log dict NOTE: this is not safe if not full steps
            log_dict = {
                'loss': np.mean(loss_dict['loss']),
                'ub_loss': np.mean(loss_dict['ub_loss']),
                'r_loss': np.mean(loss_dict['r_loss']),
                'r_ub_loss': np.mean(loss_dict['r_ub_loss']),
                'others': np.mean(loss_dict['others'], axis=0),
                'others_ub': np.mean(loss_dict['others_ub'], axis=0),
                'r_others': np.mean(loss_dict['r_others'], axis=0),
                'r_others_ub': np.mean(loss_dict['r_others_ub'], axis=0),
                'score': -1,
                'score_last': -1,
                'score_conf': -1,
                'score_last_conf': -1
            }

            # log in db
            log_log_dict(usage, log_dict)

            # print
            print_log_dict(log_dict, usage, start_time, dt, s_loss_weights, dt_s_loss_weights)


@ex.command
def run_from_file(record_grouping_score, record_relational_loss, feed_actions, run_config, nem, dt, log_dir, seed, net_path=None):
    tf.set_random_seed(seed)

    # load network weights (default is log_dir/best if net_path is not set)
    net_path = os.path.abspath(os.path.join(log_dir, 'best')) if net_path is None else net_path
    usage = run_config['usage']

    # prep weights for print out
    loss_step_weights = get_loss_step_weights()
    s_loss_weights = np.sum(loss_step_weights)
    dt_s_loss_weights = np.sum(loss_step_weights[-dt:])

    with tf.Graph().as_default() as g:
        # Set up Data
        nr_steps = nem['nr_steps'] + 1
        out_list = ['features']
        out_list.append('groups') if record_grouping_score else None
        out_list.append(record_relational_loss) if record_relational_loss else None
        out_list.append('actions') if feed_actions else None

        inputs = InputPipeLine(usage, shuffle=False, sequence_length=nr_steps, out_list=out_list, batch_size=run_config['batch_size'])

        # Build Graph
        _, _, graph, debug_graph = build_graphs(inputs.output, inputs.output)

        t = time.time()
        with tf.Session(graph=g) as session:
            coord = tf.train.Coordinator()
            saver = tf.train.Saver()
            saver.restore(session, net_path)

            # launch pipeline
            enqueue_thread = threading.Thread(target=inputs.enqueue, args=[session, coord])
            enqueue_thread.start()

            log_dict = run_epoch(session, inputs, graph, debug_graph, run_config['debug_samples'],
                                 "run_{}".format(usage))

            # log log dict
            log_log_dict(usage, log_dict)

            # shutdown pipeline
            coord.request_stop()
            session.run(inputs.queue.close(cancel_pending_enqueues=True))
            coord.join()

        print_log_dict(log_dict, usage, t, dt, s_loss_weights, dt_s_loss_weights)


@ex.automain
def run(record_grouping_score, record_relational_loss, feed_actions, net_path, training, validation, nem, dt, seed, log_dir, _run):
    save_epochs = training['save_epochs']

    # clear debug dir
    if log_dir and net_path is None:
        utils.create_directory(log_dir)
        utils.delete_files(log_dir, recursive=True)

    # prep weights for print out
    loss_step_weights = get_loss_step_weights()
    s_loss_weights = np.sum(loss_step_weights)
    dt_s_loss_weights = np.sum(loss_step_weights[-dt:])

    # Set up data pipelines
    nr_iters = nem['nr_steps'] + 1
    out_list = ['features']
    out_list.append('groups') if record_grouping_score else None
    out_list.append(record_relational_loss) if record_relational_loss else None
    out_list.append('actions') if feed_actions else None

    train_inputs = InputPipeLine('training', shuffle=True, out_list=out_list, sequence_length=nr_iters,
                                 batch_size=training['batch_size'])
    valid_inputs = InputPipeLine('validation', shuffle=False, out_list=out_list, sequence_length=nr_iters,
                                 batch_size=validation['batch_size'])

    # Build Graph
    train_op, train_graph, valid_graph, debug_graph = build_graphs(train_inputs.output, valid_inputs.output)
    init = tf.global_variables_initializer()

    # print vars
    utils.print_vars(tf.trainable_variables())

    with tf.Session() as session:
        tf.set_random_seed(seed)

        # continue training from net_path if specified
        saver = tf.train.Saver()
        if net_path is not None:
            saver.restore(session, net_path)
        else:
            session.run(init)

        # start training pipelines
        writer = tf.summary.FileWriter(log_dir, graph=session.graph,)
        coord = tf.train.Coordinator()
        train_enqueue_thread = threading.Thread(target=train_inputs.enqueue, args=[session, coord])
        coord.register_thread(train_enqueue_thread)
        train_enqueue_thread.start()
        valid_enqueue_thread = threading.Thread(target=valid_inputs.enqueue, args=[session, coord])
        coord.register_thread(valid_enqueue_thread)
        valid_enqueue_thread.start()

        best_valid_loss = np.inf
        best_valid_epoch = 0
        for epoch in range(1, training['max_epoch'] + 1):
            # run train epoch
            t = time.time()
            log_dict = run_epoch(session, train_inputs, train_graph, debug_graph, training['debug_samples'], "train_e{}".format(epoch), train_op=train_op)

            # log all items in dict
            log_log_dict('training', log_dict)

            # produce print-out
            print("\n" + 80 * "%" + "    EPOCH {}   ".format(epoch) + 80 * "%")
            print_log_dict(log_dict, 'Train', t, dt, s_loss_weights, dt_s_loss_weights)

            # run valid epoch
            t = time.time()
            log_dict = run_epoch(session, valid_inputs, valid_graph, debug_graph, validation['debug_samples'], "valid_e{}".format(epoch))

            # add logs
            log_log_dict('validation', log_dict)

            # produce plots
            create_curve_plots('loss', {'training': get_logs('training.loss'),
                                        'validation': get_logs('validation.loss')}, [0, 1000], [0, 200])
            create_curve_plots('r_loss', {'training': get_logs('training.r_loss'),
                                          'validation': get_logs('validation.r_loss')}, [0, 100], [0, 20])

            create_curve_plots('score', {'score': get_logs('validation.score'),
                                         'score_last': get_logs('validation.score_last')}, [0, 1], None)

            # produce print-out
            print("\n")
            print_log_dict(log_dict, 'Validation', t, dt, s_loss_weights, dt_s_loss_weights)

            if log_dict['loss'] < best_valid_loss:
                best_valid_loss = log_dict['loss']
                best_valid_epoch = epoch
                _run.result = float(log_dict['score']), float(log_dict['score_last']), \
                              float(log_dict['loss']), float(log_dict['ub_loss']), \
                              float(np.sum(log_dict['others'][-dt:, 1])/dt_s_loss_weights), \
                              float(np.sum(log_dict['others_ub'][-dt:, 1]) / dt_s_loss_weights), \
                              float(np.sum(log_dict['others'][-dt:, 2]) / dt_s_loss_weights), \
                              float(np.sum(log_dict['others_ub'][-dt:, 2]) / dt_s_loss_weights), \
                              float(log_dict['r_loss']), float(log_dict['r_ub_loss']), \
                              float(np.sum(log_dict['r_others'][-dt:, 1]) / dt_s_loss_weights), \
                              float(np.sum(log_dict['r_others_ub'][-dt:, 1]) / dt_s_loss_weights), \
                              float(np.sum(log_dict['r_others'][-dt:, 2]) / dt_s_loss_weights), \
                              float(np.sum(log_dict['r_others_ub'][-dt:, 2]) / dt_s_loss_weights)

                print("    Best validation loss improved to %.03f" % best_valid_loss)
                save_destination = saver.save(session, os.path.abspath(os.path.join(log_dir, 'best')))
                print("    Saved to:", save_destination)
            if epoch in save_epochs:
                save_destination = saver.save(session, os.path.abspath(os.path.join(log_dir, 'epoch_{}'.format(epoch))))
                print("    Saved to:", save_destination)

            best_valid_loss = min(best_valid_loss, log_dict['loss'])

            if best_valid_loss < np.min(get_logs('validation.loss')[-training['max_patience']:]):
                print('Early Stopping because validation loss did not improve for {} epochs'.format(training['max_patience']))
                break

            if np.isnan(log_dict['loss']):
                print('Early Stopping because validation loss is nan')
                break

        # shutdown everything to avoid zombies
        coord.request_stop()
        session.run(train_inputs.queue.close(cancel_pending_enqueues=True))
        session.run(valid_inputs.queue.close(cancel_pending_enqueues=True))
        coord.join()

    # reset the graph
    tf.reset_default_graph()

    # gather best results
    best_valid_score = float(get_logs('validation.score')[best_valid_epoch - 1])
    best_valid_score_last = float(get_logs('validation.score_last')[best_valid_epoch - 1])

    best_valid_loss = float(get_logs('validation.loss')[best_valid_epoch - 1])
    best_valid_ub_loss = float(get_logs('validation.ub_loss')[best_valid_epoch - 1])

    best_valid_intra_loss = float(np.sum(get_logs('validation.others')[best_valid_epoch - 1][-dt:, 1])/dt_s_loss_weights)
    best_valid_intra_ub_loss = float(np.sum(get_logs('validation.others_ub')[best_valid_epoch - 1][-dt:, 1])/dt_s_loss_weights)

    best_valid_inter_loss = float(np.sum(get_logs('validation.others')[best_valid_epoch - 1][-dt:, 2])/dt_s_loss_weights)
    best_valid_inter_ub_loss = float(np.sum(get_logs('validation.others_ub')[best_valid_epoch - 1][-dt:, 2])/dt_s_loss_weights)

    best_valid_r_loss = float(get_logs('validation.r_loss')[best_valid_epoch - 1])
    best_valid_r_ub_loss = float(get_logs('validation.r_ub_loss')[best_valid_epoch - 1])

    best_valid_r_intra_loss = float(np.sum(get_logs('validation.r_others')[best_valid_epoch - 1][-dt:, 1])/dt_s_loss_weights)
    best_valid_r_intra_ub_loss = float(np.sum(get_logs('validation.r_others_ub')[best_valid_epoch - 1][-dt:, 1])/dt_s_loss_weights)

    best_valid_r_inter_loss = float(np.sum(get_logs('validation.r_others')[best_valid_epoch - 1][-dt:, 2])/dt_s_loss_weights)
    best_valid_r_inter_ub_loss = float(np.sum(get_logs('validation.r_others_ub')[best_valid_epoch - 1][-dt:, 2])/dt_s_loss_weights)

    return best_valid_score, best_valid_score_last, best_valid_loss, best_valid_ub_loss, best_valid_intra_loss, \
           best_valid_intra_ub_loss, best_valid_inter_loss, best_valid_inter_ub_loss, best_valid_r_loss, \
           best_valid_r_ub_loss, best_valid_r_intra_loss, best_valid_r_intra_ub_loss, best_valid_r_inter_loss, \
           best_valid_r_inter_ub_loss
