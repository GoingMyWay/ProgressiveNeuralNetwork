#!/usr/bin/env python3
# coding: utf-8
import time
import threading

import os
import tensorflow as tf

from vizdoom import *

from a3c_agent import BasicAgent, Game
from a3c_network import BasicACNetwork, ProgNN
import configs as cfg


def main_train(tf_configs=None):
    s_t = time.time()

    tf.reset_default_graph()

    if not os.path.exists(cfg.MODEL_PATH):
        os.makedirs(cfg.MODEL_PATH)

    sess = tf.Session(config=tf_configs)

    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    with tf.device("/gpu:2"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-5, epsilon=1e-3)
        local_column_1 = BasicACNetwork('global', cfg.TASK_LIST[0], is_train=False, img_shape=cfg.IMG_SHAPE)
        local_column_2 = BasicACNetwork('global', cfg.TASK_LIST[1], is_train=True, img_shape=cfg.IMG_SHAPE)
        net_columns = [local_column_1, local_column_2]
        global_network = ProgNN(scope='global', final_task='healthpack_supreme', net_columns=net_columns, optimizer=optimizer)
        num_workers = cfg.NUM_TASKS
        agents = []
        # Create worker classes
        for i in range(num_workers):
            agents.append(
                BasicAgent(sess=sess, game=Game(cfg.SCENARIO_PATH, play=False), name=i,
                           task_list=cfg.TASK_LIST, optimizer=optimizer, global_episodes=global_episodes))
    value_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='healthpack_simple/global/networks')
    saver = tf.train.Saver(value_list, max_to_keep=100)

    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    if cfg.load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(cfg.RESTORE_MODEL_PATH)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Successfully loaded model, before spawning agents, sleep 3 seconds')
        time.sleep(3)

    saver = tf.train.Saver(max_to_keep=100)

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for ag in agents:
        t = threading.Thread(target=lambda: ag.train_a3c(sess, coord, saver))
        t.start()
        time.sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)

    sess.close()

    print("training ends, costs{}".format(time.time() - s_t))


def main_play(tf_configs=None):
    tf.reset_default_graph()

    with tf.Session(config=tf_configs) as sess:

        ag = BasicAgent(sess, Game(cfg.SCENARIO_PATH, play=True), 0, play=True)
        print('Loading Model...')
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(cfg.MODEL_PATH)
        saver.restore(sess, os.path.join(cfg.MODEL_PATH, 'model-1750.ckpt'))
        print('Successfully loaded!')

        ag.play_game(sess, 10)


if __name__ == '__main__':

    train = cfg.IS_TRAIN
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    if train:
        main_train(tf_configs=config)
    else:
        main_play(tf_configs=config)
