# coding: utf-8

s_size = 6400  # 80 * 80 * 1

img_dim = 80
a_size = 3  # LEFT, RIGHT, FORWARD
gamma = .99

load_model = True
max_episode_length = 2100
RESTORE_MODEL_PATH = './check_point/healthpack_supreme'
MODEL_PATH = 'check_point/prognn_simple_supreme'
SCENARIO_PATH = '../scenarios/health_gathering_supreme.wad'
ACTION_DIM = 2 ** a_size - 2  # remove (True, True, True) and (True, True, False)[Left, Right, Forward]

IMG_SHAPE = (80, 80)

NUM_TASKS = 16
RNN_DIM = 256

HEALTH_GATHERING_SIMPLE = 'healthpack_simple'
HEALTH_GATHERING_SUPREME = 'healthpack_supreme'
DEATH_MATCH_1 = 'death_match_1'
DEATH_MATCH_2 = 'death_match_2'

TASK_LIST = [HEALTH_GATHERING_SIMPLE, HEALTH_GATHERING_SUPREME]

TASK_NAME = HEALTH_GATHERING_SIMPLE

HIDDEN_LAYER_NUM = 4

IS_TRAIN = True
AGENTS_NUM = 16

HIST_LEN = 4

CLIP_NORM = 60.
