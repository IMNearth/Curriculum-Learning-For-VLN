from .evaluator import Evaluation
from .trainer import check_the_code, train_follower, train_selfmonitor, train_envdrop
from .curriculum import NaiveCurriculum, SelfPacedCurriculum


class ClassicTrainer(object):
    def __init__(self):pass
    
    def train(self, cfg, agent, tsboard_dir, train_env, valid_env, **kwargs):
        ''' Train the model. '''
        if cfg.MODEL.NAME == "FOLLOWER":
            return train_follower(cfg, agent, tsboard_dir, train_env, valid_env, **kwargs)
        elif cfg.MODEL.NAME == "SELF-MONITOR":
            return train_selfmonitor(cfg, agent, tsboard_dir, train_env, valid_env, **kwargs)
        elif cfg.MODEL.NAME == "ENVDROP":
            return train_envdrop(cfg, agent, tsboard_dir, train_env, valid_env, **kwargs)
        else: raise NotImplementedError

