from .base import TestAgent, BasicR2RAgent
from .follower import FollowerAgent
from .monitor import SelfMonitorAgent
# from .subinstr import SubInstructionAgent
from .envdrop import EnvDropAgent

from .vln_bert import VLNBert
from .speaker import Speaker


def build_agent(cfg, tokenizer, device, **kwargs):
    ''' Build different agent based on your settings '''
    if cfg.MODEL.NAME == "FOLLOWER":
        agent = FollowerAgent(
            model_cfg=cfg.MODEL.FOLLOWER, 
            results_dir=cfg.OUTPUT.RESULT_DIR,
            device=device, 
            env=None, # currently, do not pass any env into the model
            tokenizer=tokenizer, 
            episode_len=cfg.AGENT.MAX_EPISODE_LEN
        )
    elif cfg.MODEL.NAME == "SELF-MONITOR":
        agent = SelfMonitorAgent(
            model_cfg=cfg.MODEL.MONITOR, 
            max_enc_len=cfg.DATA.MAX_ENC_LEN, 
            results_dir=cfg.OUTPUT.RESULT_DIR,
            device=device, 
            env=None, # currently, do not pass any env into the model
            tokenizer=tokenizer, 
            episode_len=cfg.AGENT.MAX_EPISODE_LEN
        )
    # elif cfg.MODEL.NAME == "SUB-INSTR":
    #     agent = SubInstructionAgent(
    #         model_cfg=cfg.MODEL.SUB_INSTR, 
    #         max_enc_len=cfg.DATA.MAX_ENC_LEN, 
    #         max_subinstr_size=cfg.DATA.MAX_SUBINSTR_NUM, 
    #         results_dir=cfg.OUTPUT.RESULT_DIR,
    #         device=device, 
    #         env=None, # currently, do not pass any env into the model
    #         tokenizer=tokenizer, 
    #         episode_len=cfg.AGENT.MAX_EPISODE_LEN    
    #     )
    elif cfg.MODEL.NAME == "ENVDROP":
        agent = EnvDropAgent(
            model_cfg=cfg.MODEL.ENVDROP, 
            max_enc_len=cfg.DATA.MAX_ENC_LEN, 
            results_dir=cfg.OUTPUT.RESULT_DIR,
            device=device, 
            env=None, # currently, do not pass any env into the model
            tokenizer=tokenizer, 
            episode_len=cfg.AGENT.MAX_EPISODE_LEN  
        )
    else: raise NotImplementedError
    return agent


