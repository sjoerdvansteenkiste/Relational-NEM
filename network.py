#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from sacred import Ingredient
from tensorflow.contrib.rnn import RNNCell
from utils import ACTIVATION_FUNCTIONS

net = Ingredient('network')


@net.config
def cfg():
    input = [
        {'name': 'reshape', 'shape': (64, 64, 1)},
        {'name': 'conv', 'size': 16, 'act': 'elu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
        {'name': 'conv', 'size': 32, 'act': 'elu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
        {'name': 'conv', 'size': 64, 'act': 'elu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
        {'name': 'reshape', 'shape': -1},
        {'name': 'fc', 'size': 512, 'act': 'elu', 'ln': True},
    ]
    recurrent = [
        {'name': 'r_nem', 'size': 250, 'act': 'sigmoid', 'ln': True,
         'encoder': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'core': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'context': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'attention': [
             {'name': 'fc', 'size': 100, 'act': 'tanh', 'ln': True},
             {'name': 'fc', 'size': 1, 'act': 'sigmoid'},
         ]}
    ]
    output = [
        {'name': 'fc', 'size': 512, 'act': 'relu', 'ln': True},
        {'name': 'fc', 'size': 8*8*64, 'act': 'relu', 'ln': True},
        {'name': 'reshape', 'shape': (8, 8, 64)},
        {'name': 'r_conv', 'size': 32, 'act': 'relu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
        {'name': 'r_conv', 'size': 16, 'act': 'relu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
        {'name': 'r_conv', 'size': 1, 'act': 'sigmoid', 'stride': [2, 2], 'kernel': (4, 4)},
        {'name': 'reshape', 'shape': -1},
    ]

# encoder decoder pairs


net.add_named_config('enc_dec_84_atari', {
    'input': [
        {'name': 'reshape', 'shape': (84, 84, 1)},
        {'name': 'conv', 'size': 16, 'act': 'elu', 'stride': [4, 4], 'kernel': (8, 8), 'ln': True},
        {'name': 'conv', 'size': 32, 'act': 'elu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
        {'name': 'reshape', 'shape': -1},
        {'name': 'fc', 'size': 250, 'act': 'elu', 'ln': True},
    ],
    'output': [
        {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
        {'name': 'fc', 'size': 10*10*32, 'act': 'relu', 'ln': True},
        {'name': 'reshape', 'shape': (10, 10, 32)},
        {'name': 'r_conv', 'size': 16, 'act': 'relu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True, 'offset': 1},
        {'name': 'r_conv', 'size': 1, 'act': 'sigmoid', 'stride': [4, 4], 'kernel': (8, 8)},
        {'name': 'reshape', 'shape': -1},
    ]})


# recurrent configurations

net.add_named_config('rnn_250', {'recurrent': [{'name': 'rnn', 'size': 250, 'act': 'sigmoid', 'ln': True}]})
net.add_named_config('lstm_250', {'recurrent': [{'name': 'lstm', 'size': 250, 'act': 'sigmoid', 'ln': True}]})

net.add_named_config('r_nem', {
    'recurrent': [
        {'name': 'r_nem', 'size': 250, 'act': 'sigmoid', 'ln': True,
         'encoder': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'core': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'context': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'attention': [
             {'name': 'fc', 'size': 100, 'act': 'tanh', 'ln': True},
             {'name': 'fc', 'size': 1, 'act': 'sigmoid'},
         ]}
    ]})


net.add_named_config('r_nem_no_attention', {
    'recurrent': [
        {'name': 'r_nem', 'size': 250, 'act': 'sigmoid', 'ln': True,
         'encoder': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'core': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'context': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'attention': []}
    ]})


net.add_named_config('r_nem_actions', {
    'recurrent': [
        {'name': 'r_nem', 'size': 250, 'act': 'sigmoid', 'ln': True,
         'encoder': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'core': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'context': [
             {'name': 'fc', 'size': 250, 'act': 'relu', 'ln': True},
         ],
         'attention': [
             {'name': 'fc', 'size': 100, 'act': 'tanh', 'ln': True},
             {'name': 'fc', 'size': 1, 'act': 'sigmoid'},
         ],
         'actions': [
             {'name': 'fc', 'size': 10, 'act': 'relu', 'ln': True},
         ]}
    ]})


# GENERIC WRAPPERS

class InputWrapper(RNNCell):
    """Adding an input projection to the given cell."""

    def __init__(self, cell, spec, name="InputWrapper"):
        self._cell = cell
        self._spec = spec
        self._name = name

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        projected = None
        with tf.variable_scope(scope or self._name):
            if self._spec['name'] == 'fc':
                projected = slim.fully_connected(inputs, self._spec['size'], activation_fn=None)
            elif self._spec['name'] == 'conv':
                projected = slim.conv2d(inputs, self._spec['size'], self._spec['kernel'], self._spec['stride'], activation_fn=None)
            else:
                raise ValueError('Unknown layer name "{}"'.format(self._spec['name']))

        return self._cell(projected, state)


class OutputWrapper(RNNCell):
    """Adding an output projection to the given cell."""

    def __init__(self, cell, spec, n_out=1, name="OutputWrapper"):
        self._cell = cell
        self._spec = spec
        self._name = name
        self._n_out = n_out

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._spec['size']

    def __call__(self, inputs, state, scope=None):
        output, res_state = self._cell(inputs, state)

        projected = None
        with tf.variable_scope((scope or self._name)):
            if self._spec['name'] == 'fc':
                projected = slim.fully_connected(output, self._spec['size'], activation_fn=None)
            elif self._spec['name'] == 'r_conv':
                offset = self._spec.get('offset', 0)
                resized = tf.image.resize_images(output, (self._spec['stride'][0] * output.get_shape()[1].value + offset,
                                                          self._spec['stride'][1] * output.get_shape()[2].value + offset), method=1)
                projected = slim.layers.conv2d(resized, self._spec['size'], self._spec['kernel'], activation_fn=None)
            else:
                raise ValueError('Unknown layer name "{}"'.format(self._spec['name']))

        return projected, res_state


class ReshapeWrapper(RNNCell):
    def __init__(self, cell, shape='flatten', apply_to='output'):
        self._cell = cell
        self._shape = shape
        self._apply_to = apply_to

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        batch_size = tf.shape(inputs)[0]

        if self._apply_to == 'input':
            inputs = slim.flatten(inputs) if self._shape == -1 else tf.reshape(inputs, [batch_size] + self._shape)
            return self._cell(inputs, state)
        elif self._apply_to == 'output':
            output, res_state = self._cell(inputs, state)
            output = slim.flatten(output) if self._shape == -1 else tf.reshape(output, [batch_size] + self._shape)
            return output, res_state
        elif self._apply_to == 'state':
            output, res_state = self._cell(inputs, state)
            res_state = slim.flatten(res_state) if self._shape == -1 else tf.reshape(res_state, [batch_size] + self._shape)
            return output, res_state
        else:
            raise ValueError('Unknown apply_to: "{}"'.format(self._apply_to))


class ActivationFunctionWrapper(RNNCell):
    def __init__(self, cell, activation='linear', apply_to='output'):
        self._cell = cell
        self._activation = ACTIVATION_FUNCTIONS[activation]
        self._apply_to = apply_to

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        if self._apply_to == 'input':
            inputs = self._activation(inputs)
            return self._cell(inputs, state)
        elif self._apply_to == 'output':
            output, res_state = self._cell(inputs, state)
            output = self._activation(output)
            return output, res_state
        elif self._apply_to == 'state':
            output, res_state = self._cell(inputs, state)
            res_state = self._activation(res_state)
            return output, res_state
        else:
            raise ValueError('Unknown apply_to: "{}"'.format(self._apply_to))


class LayerNormWrapper(RNNCell):
    def __init__(self, cell, apply_to='output', name="LayerNorm"):
        self._cell = cell
        self._name = name
        self._apply_to = apply_to

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        if self._apply_to == 'input':
            with tf.variable_scope(scope or self._name):
                inputs = slim.layer_norm(inputs)
            return self._cell(inputs, state)
        elif self._apply_to == 'output':
            output, res_state = self._cell(inputs, state)
            with tf.variable_scope(scope or self._name):
                output = slim.layer_norm(output)
                return output, res_state
        elif self._apply_to == 'state':
            output, res_state = self._cell(inputs, state)
            with tf.variable_scope(scope or self._name):
                res_state = slim.layer_norm(res_state)
                return output, res_state
        else:
            raise ValueError('Unknown apply_to: "{}"'.format(self._apply_to))
            

# R-NEM CELL

class R_NEM(RNNCell):
    def __init__(self, encoder, core, context, attention, actions, size, K, name='NPE'):
        self._encoder = encoder
        self._core = core
        self._context = context
        self._attention = attention
        self._actions = actions

        assert K > 1
        self._size = size
        self._K = K
        self._name = name

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def get_shapes(self, inputs):
        bk = tf.shape(inputs)[0]
        m = tf.shape(inputs)[1]

        return bk // self._K, self._K, m

    def __call__(self, inputs, state, scope=None):
        """
        input: [B X K, M]
        state: [B x K, H]

        b: batch_size
        k: num_groups
        m: input_size
        h: hidden_size
        h1: size of the encoding of focus and context
        h2: size of effect
        o: size of output

        # 0. Encode with RNN: x is [B*K, M], h is [B*K, H] --> both are [B*K, H]
        # 1. Reshape both to [B, K, H]
        # 2. For each of the k \in K copies, extract the K-1 states that are not that k
        # 3. Now you have two tensors of size [B x K x K-1, H]
        #     The first: "focus object": K-1 copies of the state of "k", the focus object
        #     The second: "context objects": K-1 (all unique) states of the context objects
        # 4. Concatenate results of 3
        # 5. Core: Process result of 4 in a feedforward network --> [B x K, H'']
        # 6. Reshape to [B x K, K-1, H''] to isolate the K-1 dimension (because we did for K-1 pairs)
        # 7. Sum in the K-1 dimension --> [B x K, H'']
        #   7.5 weighted by attention
        # 8. Decoder: Concatenate result of 7, the original theta, and the x and process into new state --> [B x K, H]
        # 9. Actions: Optionally embed actions into some representation

        """
        with tf.variable_scope(scope or self._name):
            b, k, m = self.get_shapes(inputs)

            # compute action embedding and concat to state
            if self._actions:
                action = state['action']
                state = state['state']

                # Optionally compute actions
                action_embedding = action[:, 0]

                for i, layer in enumerate(self._actions):
                    action_embedding = self._build_layer(action_embedding, layer)

                action_embedding = tf.tile(action_embedding, [k, 1])  # (b * k, <embed_size>)

                # concat to current state size
                state = tf.concat((state, action_embedding), axis=1)

            # Encode theta
            state1 = state
            for i, layer in enumerate(self._encoder):
                state1 = self._build_layer(state1, layer)

            # Reshape theta to be used for context
            h1 = state1.get_shape().as_list()[1]
            state1r = tf.reshape(state1, [b, k, h1])     # (b, k, h1)

            # Reshape theta to be used for focus
            state1rr = tf.reshape(state1r, [b, k, 1, h1])     # (b, k, 1, h1)

            # Create focus: tile state1rr k-1 times
            fs = tf.tile(state1rr, [1, 1, k-1, 1])   # (b, k, k-1, h1) 

            # Create context
            state1rl = tf.unstack(state1r, axis=1)      # list of length k of (b, h1)

            if k > 1:
                csu = []
                for i in range(k):
                    selector = [j for j in range(k) if j != i]
                    c = list(np.take(state1rl, selector))  # list of length k-1 of (b, h1)
                    c = tf.stack(c, axis=1)     # (b, k-1, h1)
                    csu.append(c)

                cs = tf.stack(csu, axis=1)    # (b, k, k-1, h1)   
            else:
                cs = tf.zeros((b, k, k-1, h1))

            # Reshape focus and context 
            # you will process the k-1 instances through the same network anyways
            fsr, csr = tf.reshape(fs, [b*k*(k-1), h1]), tf.reshape(cs, [b*k*(k-1), h1])     # (b x k x k-1, h1)

            # Concatenate focus and context
            concat = tf.concat([fsr, csr], axis=1)    # (b x k x k-1, 2h1)

            # NPE core
            core_out = concat
            for i, layer in enumerate(self._core):
                core_out = self._build_layer(core_out, layer)

            # Context branch: produces context
            context = core_out
            for i, layer in enumerate(self._context):
                context = self._build_layer(context, layer)

            h2 = self._context[-1]['size'] if len(self._context) > 0 else self._core[-1]['size']
            contextr = tf.reshape(context, [b*k, k-1, h2])    # (b x k, k-1, h2)

            # Attention branch: produces attention coefficients
            if len(self._attention) > 0:
                attention = core_out

                for i, layer in enumerate(self._attention):
                    attention = self._build_layer(attention, layer)

            # produce effect as sum(context * attention)
            # if len(self._attention) > 0:
                attentionr = tf.reshape(attention, [b*k, k-1, 1])
                effectrsum = tf.reduce_sum(contextr * attentionr, axis=1)
            else:
                effectrsum = tf.reduce_sum(contextr, axis=1)

            # 9 calculate new state
            # This is where the input from the encoder comes in 
            # concatenate state1, effectrsum, and input
            if self._actions:
                total = tf.concat([state1, effectrsum, inputs, action_embedding], axis=1)
            else:
                total = tf.concat([state1, effectrsum, inputs], axis=1)     # (b x k, h + h2 + m)

            # produce recurrent update
            new_state = slim.fully_connected(total, self._size, activation_fn=None)  # (b x k, h)

            return new_state, new_state

    @staticmethod
    def _build_layer(inputs, layer):
        # apply transformation
        if layer['name'] == 'fc':
            out = slim.fully_connected(inputs, layer['size'], activation_fn=None)
        else:
            raise KeyError('Unknown layer "{}"'.format(layer['name']))

        # apply layer normalisation
        if layer.get('ln', False):
            out = slim.layer_norm(out)

        # apply activation function
        if layer.get('act', False):
            out = ACTIVATION_FUNCTIONS[layer['act']](out)

        return out


# NETWORK BUILDER

@net.capture
def build_network(K, input, recurrent, output):
    with tf.name_scope('inner_RNN'):
        # build recurrent
        for i, layer in enumerate(recurrent):
            if layer['name'] == 'rnn':
                cell = tf.contrib.rnn.BasicRNNCell(layer['size'], activation=ACTIVATION_FUNCTIONS['linear'])
                cell = LayerNormWrapper(cell, apply_to='output', name='LayerNormR{}'.format(i)) if layer.get('ln') else cell
                cell = ActivationFunctionWrapper(cell, activation=layer['act'], apply_to='state')
                cell = ActivationFunctionWrapper(cell, activation=layer['act'], apply_to='output')

            elif layer['name'] == 'lstm':
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(layer['size'], layer_norm=layer.get('ln', False))

                if layer.get('act'):
                    print("WARNING: activation function arg for LSTM Cell is ignored. Default (tanh) is used in stead.")

            elif layer['name'] == 'r_nem':
                cell = R_NEM(encoder=layer['encoder'],
                             core=layer['core'],
                             context=layer['context'],
                             attention=layer['attention'],
                             actions=layer.get('actions', None),
                             size=layer['size'],
                             K=K)

                cell = LayerNormWrapper(cell, apply_to='output', name='LayerNormR{}'.format(i)) if layer.get('ln') else cell
                cell = ActivationFunctionWrapper(cell, activation=layer['act'], apply_to='state')
                cell = ActivationFunctionWrapper(cell, activation=layer['act'], apply_to='output')
            else:
                raise ValueError('Unknown recurrent name "{}"'.format(layer['name']))

        # build input
        for i, layer in reversed(list(enumerate(input))):
            if layer['name'] == 'reshape':
                cell = ReshapeWrapper(cell, layer['shape'], apply_to='input')
            else:
                cell = ActivationFunctionWrapper(cell, layer['act'], apply_to='input')
                cell = LayerNormWrapper(cell, apply_to='input', name='LayerNormI{}'.format(i)) if layer.get('ln') else cell
                cell = InputWrapper(cell, layer, name="InputWrapper{}".format(i))

        # build output
        for i, layer in enumerate(output):
            if layer['name'] == 'reshape':
                cell = ReshapeWrapper(cell, layer['shape'])
            else:
                n_out = layer.get('n_out', 1)
                cell = OutputWrapper(cell, layer, n_out=n_out, name="OutputWrapper{}".format(i))
                cell = LayerNormWrapper(cell, apply_to='output', name='LayerNormO{}'.format(i)) if layer.get('ln') else cell
                cell = ActivationFunctionWrapper(cell, layer['act'], apply_to='output')

        return cell
