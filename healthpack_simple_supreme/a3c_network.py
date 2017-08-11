# coding: utf-8
# implement neural network here
import random

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import utils
import configs as cfg


DIM = 128
LSTM_CELL_NUM = 200


class BasicACNetwork(object):
    """
    Actor-Critic network
    """
    def __init__(self, scope, task_name, is_train=False, img_shape=(80, 80)):
        self.scope = scope
        self.is_train = is_train
        self.task_name = task_name
        self.__create_network(scope, img_shape=img_shape)
        self.__init_layers()

    def __create_network(self, scope, img_shape=(80, 80)):
        with tf.variable_scope(self.task_name):
            with tf.variable_scope(scope):
                with tf.variable_scope('input_data'):
                    self.inputs = tf.placeholder(shape=[None, *img_shape, cfg.HIST_LEN], dtype=tf.float32)
                with tf.variable_scope('networks'):
                    with tf.variable_scope('conv_1'):
                        self.conv_1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.inputs, num_outputs=32,
                                                  kernel_size=[8, 8], stride=4, padding='SAME', trainable=self.is_train)
                    with tf.variable_scope('conv_2'):
                        self.conv_2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv_1, num_outputs=64,
                                                  kernel_size=[4, 4], stride=2, padding='SAME', trainable=self.is_train)
                    with tf.variable_scope('conv_3'):
                        self.conv_3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv_2, num_outputs=64,
                                                  kernel_size=[3, 3], stride=1, padding='SAME', trainable=self.is_train)
                    with tf.variable_scope('f_c'):
                        self.fc = slim.fully_connected(slim.flatten(self.conv_3), 512,
                                                       activation_fn=tf.nn.elu, trainable=self.is_train)

    def __init_layers(self):
        self.layers = [
            self.conv_1,
            self.conv_2,
            self.conv_3,
            self.fc
        ]
        self.net_topology = [
            [[4, 4], 2],
            [[3, 3], 1],
            512
        ]


class ProgNN(object):
    """
    Progressive Neural Network
    --------------------------
    
    One task(scenario) one column
    """
    def __init__(self, scope, net_columns, optimizer, final_task, play=False):
        if not isinstance(optimizer, tf.train.Optimizer):
            raise TypeError('the type of optimizer must be tf.train.Optimizer')
        for col in net_columns:
            if not isinstance(col, BasicACNetwork):
                raise TypeError('the type of column must be BasicACNetwork')
        self.optimizer = optimizer
        self.scope = scope
        self.final_task = final_task
        self.play = play
        self.output_column = []
        self.columns = net_columns  # list of networks
        self.hidden_layer_num = cfg.HIDDEN_LAYER_NUM
        self.assemble_network(final_task=self.final_task)

    def assemble_network(self, final_task):
        # append the first hidden layer of the last column
        with tf.variable_scope(final_task):
            with tf.variable_scope(self.scope):
                self.output_column.append(self.columns[1].layers[0])
                for _l in range(1, self.hidden_layer_num):  # from the second hidden layer
                    for _c in range(1, len(self.columns)):  # from the second column
                        pre_col, col = self.columns[_c-1], self.columns[_c]

                        a = tf.get_variable(name="adapter_%s_%s_%s" % (_l-1, _l, _c),
                                            shape=[1],
                                            initializer=tf.constant_initializer(0.01),
                                            trainable=True)
                        a_h = tf.multiply(a, pre_col.layers[_l-1], name='ah_%s_%s_%s' % (_l-1, _l, _c))
                        map_in = a_h.get_shape().as_list()[3]
                        map_out = int(map_in / (2 * _c))
                        V = tf.get_variable(name='conv2d_V_%s_%s_%s' % (_l-1, _l, _c),
                                            shape=[1, 1, map_in, map_out],
                                            initializer=utils.xavier_initializer_conv2d(),
                                            trainable=True)
                        b = tf.get_variable(name='conv2d_Vb_%s_%s_%s' % (_l-1, _l, _c),
                                            shape=[map_out],
                                            initializer=tf.constant_initializer(0),
                                            trainable=True)

                        V_a_h = tf.nn.relu(tf.add(tf.nn.conv2d(a_h, V, strides=[1, 1, 1, 1], padding='SAME'), b),
                                           name='V_a_h_%s_%s_%s' % (_l-1, _l, _c))

                        if _l != self.hidden_layer_num - 1:  # conv -> conv, last layer

                            size, stride = pre_col.net_topology[_l-1]
                            in_size, out_size = V_a_h.get_shape().as_list()[3], col.layers[_l].get_shape().as_list()[3]
                            U = tf.get_variable(name='conv2d_U_%s_%s_%s' % (_l-1, _l, _c),
                                                shape=[*size, in_size, out_size],
                                                initializer=utils.xavier_initializer_conv2d(),
                                                trainable=True)
                            b = tf.get_variable(name='conv2d_Ub_%s_%s_%s' % (_l-1, _l, _c),
                                                shape=[out_size],
                                                initializer=tf.constant_initializer(0),
                                                trainable=True)
                            U_V_a_h = tf.nn.relu(
                                tf.add(tf.nn.conv2d(V_a_h, U, strides=[1, stride, stride, 1], padding='SAME'), b),
                                name='U_V_a_h_%s_%s_%s' % (_l-1, _l, _c))
                        else:  # conv -> fc
                            input_size = utils.reduce_multiply(V_a_h.get_shape().as_list()[1:])
                            neuron_size = pre_col.net_topology[_l-1]
                            U = tf.get_variable(name='fc_U_%s_%s_%s' % (_l-1, _l, _c),
                                                shape=[input_size, neuron_size],
                                                initializer=utils.xavier_initializer_conv2d(),
                                                trainable=True)
                            b = tf.get_variable(name='fc_Ub_%s_%s_%s' % (_l-1, _l, _c),
                                                shape=[neuron_size],
                                                initializer=utils.xavier_initializer_conv2d(),
                                                trainable=True)
                            U_V_a_h = \
                                tf.add(tf.matmul(slim.flatten(V_a_h), U), b, name='U_V_a_h_%s_%s_%s' % (_l-1, _l, _c))

                        output = tf.nn.relu(col.layers[_l] + U_V_a_h, name='output_%s_%s_%s' % (_l-1, _l, _c))
                        self.output_column.append(output)

                with tf.variable_scope('pnn_actor_critic'):
                    with tf.variable_scope('pnn_actor'):
                        self.policy = slim.fully_connected(self.output_column[-1],
                                                           cfg.ACTION_DIM,
                                                           activation_fn=tf.nn.softmax,
                                                           biases_initializer=None)
                    with tf.variable_scope('pnn_critic'):
                        self.value = slim.fully_connected(self.output_column[-1],
                                                          1,
                                                          activation_fn=None,
                                                          biases_initializer=None)

        if self.scope != 'global' and not self.play:
            with tf.variable_scope('pnn'):
                with tf.variable_scope('action_input'):
                    self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                    self.actions_onehot = tf.one_hot(self.actions, cfg.ACTION_DIM, dtype=tf.float32)
                with tf.variable_scope('target_v'):
                    self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                with tf.variable_scope('advantage'):
                    self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, axis=1)
                with tf.variable_scope('loss_func'):
                    with tf.variable_scope('policy_loss'):
                        self.policy_loss = -tf.reduce_sum(self.advantages * tf.log(self.responsible_outputs+1e-10))
                    with tf.variable_scope('value_loss'):
                        self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                    with tf.variable_scope('entropy_loss'):
                        self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy+1e-10))
                    with tf.variable_scope('a3c_loss'):
                        self.loss = self.policy_loss + 0.5 * self.value_loss - 0.005 * self.entropy

                with tf.variable_scope('asynchronize'):
                    with tf.variable_scope('get_local_grad'):
                        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.final_task+'/'+self.scope)
                        self.gradients = tf.gradients(self.loss, local_vars)
                        self.var_norms = tf.global_norm(local_vars)
                        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, cfg.CLIP_NORM)

                    with tf.variable_scope('push'):
                        push_global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, final_task+'/global')
                        self.apply_grads = self.optimizer.apply_gradients(zip(grads, push_global_vars))

                    with tf.variable_scope('pull'):
                        pull_global_vars = tf.get_collection(
                                                tf.GraphKeys.TRAINABLE_VARIABLES, final_task+'/global')
                        update_local_vars = tf.get_collection(
                                                tf.GraphKeys.TRAINABLE_VARIABLES, self.final_task+'/'+self.scope)
                        self.pull_assign_value = [l.assign(g) for g, l in zip(pull_global_vars, update_local_vars)]

    def pull(self, session):
        if not isinstance(session, tf.Session):
            raise TypeError('Invalid Type')

        session.run(self.pull_assign_value)

    def push(self, session, feed_dict):
        if not isinstance(session, tf.Session):
            raise TypeError('Invalid Type')

        session.run(self.apply_grads, feed_dict)

    def get_action_index_and_value(self, session, feed_dict, deterministic=False):
        if not isinstance(session, tf.Session) and not isinstance(feed_dict, dict):
            raise TypeError('Invalid type')

        action_dist, value = session.run([self.policy, self.value], feed_dict)
        a_index = self.choose_action_index(action_dist[0], deterministic=deterministic)
        return a_index, value[0, 0]

    def get_action_index(self, session, feed_dict, deterministic=False):
        if not isinstance(session, tf.Session) and not isinstance(feed_dict, dict):
            raise TypeError('Invalid type')

        action_dist = session.run(self.policy, feed_dict)
        a_index = self.choose_action_index(action_dist[0], deterministic=deterministic)
        return a_index

    def get_value(self, session, feed_dict):
        if not isinstance(session, tf.Session) and not isinstance(feed_dict, dict):
            raise TypeError('Invalid type')

        return session.run(self.value, feed_dict)[0, 0]

    @staticmethod
    def choose_action_index(policy, deterministic=False):
        if deterministic:
            return np.argmax(policy)

        r = random.random()
        cumulative_reward = 0
        for i, p in enumerate(policy):
            cumulative_reward += p
            if r <= cumulative_reward:
                return i

        return len(policy) - 1
