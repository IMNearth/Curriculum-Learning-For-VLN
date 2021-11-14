from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()                      #  ---- data options ---- 
_C.DATA.NAME = 'R2R'                # dataset name, R2R / FGR2R
_C.DATA.DATA_DIR = 'tasks/R2R-judy/data'  # where to load the dataset
_C.DATA.TRAIN_VOCAB = ''            # path to train vocab
_C.DATA.TRAINVAL_VOCAB = ''         # path to train and validation vocab
_C.DATA.MAX_ENC_LEN = 20            # maximun allowed instuction length to be encoded
_C.DATA.MAX_SUBINSTR_NUM = 0        # maximum number of sub-instrcutions given a single instruction
_C.DATA.IMG_FEAT_DIR = ''           # path to pre-cached image features


_C.TRAIN = CN()                     #  ---- training configurations  ---- 
_C.TRAIN.DEVICE = 0                 # gpu id, if no gpu is available, automatically switch to cpu
_C.TRAIN.OPTIM = ''                 # optimizer used to train the model, rmsprop / adam / sgd
_C.TRAIN.LR = 0.0001                # learning rate
_C.TRAIN.BATCH_SIZE = 128           # batch size
_C.TRAIN.START_EPOCH = 1            # useful when you resume training, default is 1
_C.TRAIN.MAX_EPOCH = 0              # maximun epoch allowed
_C.TRAIN.ITER_PER_EPOCH = 200       # number of iterations per epoch
_C.TRAIN.EVAL_INTERVAL = 1          # how often do we eval the trained model
_C.TRAIN.SCHEDULER = ''             # learning rate scheduler
_C.TRAIN.PATIENCE = 3               # Number of epochs with no improvement after which learning rate will be reduced.
_C.TRAIN.LR_MIN = 1e-6              # A lower bound on the learning rate of all param groups
_C.TRAIN.DATA_ARGUMENT = False      # whther to use data argumentation
_C.TRAIN.PROGMONITOR_WEIGHT = 0.5   # self-monitoring agent, weight of the progress loss
_C.TRAIN.EVAL_TRAIN = False         # whether to evaluate the performance on trainset

_C.TRAIN.CLMODE = ""                # wehter to use curriculum training, and which mode to use, "NAIVE | SELF-PACE"

_C.TRAIN.SELF_PACE = CN()            # -- Self-Paced Curriculum Learning (SPCL) --
_C.TRAIN.SELF_PACE.CRATE = 1.0       # how to generate constant c
_C.TRAIN.SELF_PACE.WCTRL = 0.0       # weight control variable, 0.0 ~ 1.0
_C.TRAIN.SELF_PACE.LAMB = .0         # lambda
_C.TRAIN.SELF_PACE.MIU = .0          # miu, stepsize
_C.TRAIN.SELF_PACE.FUNC = ""         # pacing function
_C.TRAIN.SELF_PACE.BURN_IN = 0       # number of burn-in epochs
_C.TRAIN.SELF_PACE.INTERVAL = 0      # update interval
_C.TRAIN.SELF_PACE.STRATEGY = ""     # update strategy

_C.TRAIN.AUTO_CULM = CN()            # -- Automated Curriculum Learning (ACL) --
_C.TRAIN.AUTO_CULM.ALPHA = .0        # parameter for Exp3.S algo
_C.TRAIN.AUTO_CULM.ETA = .0          # parameter for Exp3.S algo
_C.TRAIN.AUTO_CULM.BETA = .0         # parameter for Exp3.S algo
_C.TRAIN.AUTO_CULM.EPS = .0          # parameter for Exp3.S algo
_C.TRAIN.AUTO_CULM.RRSIZE = 0        # parameter for Exp3.S algo


_C.OUTPUT = CN()                    # ---- output options ---- 
_C.OUTPUT.RESUME = ''               # two options for resuming the model: latest | best
_C.OUTPUT.CKPT_DIR = ''             # the directory to save trained models (namely checkpoints)
_C.OUTPUT.LOG_DIR = ''              # the directory to tensorboard log files (.log and .events)
_C.OUTPUT.RESULT_DIR = ''           # the directory to model generated trajectories
_C.OUTPUT.TSBOARD = 1               # use TensorBoard for loss visualization
_C.OUTPUT.TSBOARD_DIR = ''          # the directory to save the tensorboard files


_C.AGENT = CN()                     # ---- agent options ---- 
_C.AGENT.TEACHER_FORCE = False      # follow the shortest path to the goal, by default we use student forcing
_C.AGENT.MAX_EPISODE_LEN = 20       # in a single episode, the maximun length an agent allowed to run
_C.AGENT.FEEDBACK = "sample"        # options: sample (both mode) | argmax (for testing only), note that argmax is student forcing


_C.MODEL = CN()                     # ---- model options ---- 
_C.MODEL.NAME = ""                  # which model to use

_C.MODEL.FOLLOWER = CN()                  # -- Follower -- 
_C.MODEL.FOLLOWER.GLOVE_PATH = ""         # path to glove embeddings
_C.MODEL.FOLLOWER.WORD_EMB_SIZE = 0       # for init the embedding table in encoder
_C.MODEL.FOLLOWER.HIDDEN_SIZE = 0         # hidden layer size for encoder + decoder
_C.MODEL.FOLLOWER.DROP_RATE = 0.5         # dropout rate
_C.MODEL.FOLLOWER.ENC_BIDIRECTION = True  # encoder - bi-directional
_C.MODEL.FOLLOWER.ENC_LAYERS = 1          # encoder - number of layers

_C.MODEL.MONITOR = CN()                   # -- Self-Monitoring -- 
_C.MODEL.MONITOR.WORD_EMB_SIZE = 0        # for init the embedding table in encoder
_C.MODEL.MONITOR.HIDDEN_SIZE = 0          # hidden layer size for encoder + decoder
_C.MODEL.MONITOR.DROP_RATE = 0.5          # dropout rate
_C.MODEL.MONITOR.ENC_BIDIRECTION = True   # encoder - bi-directional
_C.MODEL.MONITOR.ENC_LAYERS = 1           # encoder - number of layers
_C.MODEL.MONITOR.MLP_HIDDEN = (128, )     # policy module - hidden_dims in mlp layer

_C.MODEL.ENVDROP = CN()                   # -- Backtranslation with Environmental Dropout --
_C.MODEL.ENVDROP.WORD_EMB_SIZE = 0        # for init the embedding table in encoder
_C.MODEL.ENVDROP.ACT_EMB_SIZE = 0         # action embedding size
_C.MODEL.ENVDROP.HIDDEN_SIZE = 0          # hidden layer size for encoder + decoder
_C.MODEL.ENVDROP.DROP_RATE = 0.5          # dropout rate
_C.MODEL.ENVDROP.FEAT_DROP_RATE = 0.3     # image feature drop rate with model
_C.MODEL.ENVDROP.ENC_BIDIRECTION = True   # encoder - bi-directional
_C.MODEL.ENVDROP.ENC_LAYERS = 1           # encoder - number of layers
_C.MODEL.ENVDROP.ML_WEIGHT = 0.0          # ml - weight for cross entropy, or imitation learning
_C.MODEL.ENVDROP.GAMMA = 0.0              # rl - reward discounting factor
_C.MODEL.ENVDROP.RL_NORMALIZE = 'none'    # rl - normalize weight for rl_loss, "total|batch|none"

_C.MODEL.SUB_INSTR = CN()                   # -- Sub Instruction Aware VLN -- 
_C.MODEL.SUB_INSTR.WORD_EMB_SIZE = 0        # for init the embedding table in encoder
_C.MODEL.SUB_INSTR.HIDDEN_SIZE = 0          # hidden layer size for encoder + decoder
_C.MODEL.SUB_INSTR.DROP_RATE = 0.5          # dropout rate
_C.MODEL.SUB_INSTR.ENC_BIDIRECTION = True   # encoder - bi-directional
_C.MODEL.SUB_INSTR.ENC_LAYERS = 1           # encoder - number of layers
_C.MODEL.SUB_INSTR.MLP_HIDDEN = (128, )     # policy module - hidden_dims in mlp layer
_C.MODEL.SUB_INSTR.SHIFT_HIDDEN = 0         # shift module - hidden size


_C.AIDE = CN()                      # ---- navigation assistance options ---- 

_C.AIDE.SPEAKER = CN()                  # -- Speaker --
_C.AIDE.SPEAKER.RNN_DIM = 512           # dimension of rnn
_C.AIDE.SPEAKER.DROPOUT = 0.6           # drop out rate
_C.AIDE.SPEAKER.FEAT_DROPOUT = 0.3      # img feature drop out
_C.AIDE.SPEAKER.BI_DIRECTION = True     # use bidirectional rnn or not
_C.AIDE.SPEAKER.WEMB = 256              # word embedding size
_C.AIDE.SPEAKER.LR = 1e-4               # learning rate for speaker (not used in this code)
_C.AIDE.SPEAKER.FAST_TRAIN = False      # fast train or nor (not used in this code)
_C.AIDE.SPEAKER.IGNORE_ID = -1          # action index that should be ignored, should be -1
_C.AIDE.SPEAKER.MAX_DECODE = 120        # max output instruction length
_C.AIDE.SPEAKER.LOAD_OPTIM = False      # whether to load optimizer state_dict


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()