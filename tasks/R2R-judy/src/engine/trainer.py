import os
import os.path as osp
import time
import logging
logger = logging.getLogger("main.train")

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

import src.utils as utils
from src.environ import R2RBatch
from src.agent import TestAgent, BasicR2RAgent
from .evaluator import Evaluation

optim_switcher = {
    "adam" : torch.optim.Adam, 
    "rms" : torch.optim.RMSprop, 
    "sgd" : torch.optim.SGD,
}

###############################
#       Sanity Check          #
###############################

def check_the_code(cfg, device, tok, valid_env):
    agent = TestAgent(
        results_dir=cfg.OUTPUT.RESULT_DIR, 
        device=device, 
        env=valid_env['val_unseen'],
        tokenizer=tok, 
        episode_len=cfg.AGENT.MAX_EPISODE_LEN
    )
    
    agent.test()
    evaluator = Evaluation(splits=['val_unseen'])
    score_summary, __ = evaluator.score(agent.get_results())
    utils.prettyprint({'val_unseen': score_summary})


###############################
#         Follower            #
###############################

def train_follower(cfg, agent:BasicR2RAgent, tsboard_dir:str, train_env:R2RBatch, valid_env:dict, 
    eval_train:bool=False, eval_train_env:dict=None,
):
    ''' train the agent '''
    time_str = time.strftime('%Y-%m%d-%H:%M', time.localtime(time.time()))
    summary_save_dir = os.path.join(tsboard_dir, time_str)
    writer = SummaryWriter(logdir=summary_save_dir)
    print(f"\t You can find tensorboard summary at {summary_save_dir}.")
    logger.info(f"\t You can find tensorboard summary at {summary_save_dir}.")

    train_cfg = cfg.TRAIN

    if cfg.OUTPUT.RESUME != "":
        print(f"\t LOAD the {cfg.MODEL.NAME} model from {cfg.OUTPUT.RESUME} ...")
        logger.info(f"\t LOAD the {cfg.MODEL.NAME} model from {cfg.OUTPUT.RESUME} ...")
        ckpt = agent.load_model(osp.join(cfg.OUTPUT.CKPT_DIR, f"{cfg.OUTPUT.RESUME}.pt"))
        if "train_cfg" in ckpt: train_cfg = ckpt["train_cfg"]
        if "last_epoch" in ckpt: train_cfg.START_EPOCH = ckpt["last_epoch"]

    which_optim = optim_switcher.get(train_cfg.OPTIM, torch.optim.Adam)
    encoder_optimer = which_optim(agent.encoder.parameters(), lr=train_cfg.LR)
    decoder_optimer = which_optim(agent.decoder.parameters(), lr=train_cfg.LR)

    if cfg.DATA.NAME != "RxR":
        valid_evaluator = {
            "val_unseen": Evaluation(splits=['val_unseen']), 
            "val_seen": Evaluation(splits=['val_seen'])
        }
    else:
        valid_evaluator = {
            "val_unseen": Evaluation(splits=['val_unseen'], data_name=cfg.DATA.NAME, data_dir=cfg.DATA.DATA_DIR), 
            "val_seen": Evaluation(splits=['val_seen'], data_name=cfg.DATA.NAME, data_dir=cfg.DATA.DATA_DIR)
        }
    best_val = {'val_seen': {"success_rate": 0.}, 'val_unseen': {"success_rate": 0.}}

    if eval_train:
        train_evaluator = {}
        for k in range(1, 6):
            train_evaluator[f"round_{k}"] = Evaluation(
                [f'train_round[{k}]_v3'], data_name="CLR2R", data_dir="tasks/R2R-judy/data/CLR2Rv3")
        assert eval_train_env is not None, "Please give the eval environment!"
        assert eval_train_env.keys() == train_evaluator.keys(), "Key not Match!"

    output_ckpt_dir = os.path.join(cfg.OUTPUT.CKPT_DIR, time_str)
    if not os.path.exists(output_ckpt_dir): os.makedirs(output_ckpt_dir)
    print(f"\t You can find checkpoints at {output_ckpt_dir}.")
    logger.info(f"\t You can find checkpoints at {output_ckpt_dir}.")

    start_time, last_time = time.time(), time.time()

    for ep in range(train_cfg.START_EPOCH, train_cfg.MAX_EPOCH+1):

        agent.env = train_env
        agent.train()
        agent.losses = []
        
        for __ in range(1, train_cfg.ITER_PER_EPOCH+1):
            encoder_optimer.zero_grad()
            decoder_optimer.zero_grad()

            agent.rollout(train_ml=True, feedback=cfg.AGENT.FEEDBACK)
            agent.ml_loss.backward()

            # torch.nn.utils.clip_grad_norm(agent.encoder.parameters(), 40.)
            # torch.nn.utils.clip_grad_norm(agent.decoder.parameters(), 40.)

            encoder_optimer.step()
            decoder_optimer.step()
        
        # recoding training infomation
        epoch_loss = sum(agent.losses)
        avg_iter_loss = epoch_loss / len(agent.losses)
        max_iter_loss = max(agent.losses)
        min_iter_loss = min(agent.losses)
        writer.add_scalar("train/ml_epoch", epoch_loss, ep)
        writer.add_scalar("train/ml_iter_avg", avg_iter_loss, ep)
        writer.add_scalar("train/ml_iter_max", max_iter_loss, ep)
        writer.add_scalar("train/ml_iter_min", min_iter_loss, ep)

        # routine print
        epoch_time_cost = (time.time()-last_time)/60
        remaining_time_cost = ((time.time()-start_time)/(60 * (ep+1-train_cfg.START_EPOCH))) * (train_cfg.MAX_EPOCH - ep)
        train_print_str = f"\t Epoch [{ep}/{train_cfg.MAX_EPOCH}], Time Cost {epoch_time_cost:.2f}min/ep," + \
                          f" Remaining {remaining_time_cost:.2f} min, Train Loss: {epoch_loss:.4f}," + \
                          f" Iter [AVG: {avg_iter_loss:.4f}, MIN: {min_iter_loss:.4f}, MAX: {max_iter_loss:.4f}],"
        print(train_print_str)
        logger.info(train_print_str)

        # evaluation
        if ep % train_cfg.EVAL_INTERVAL == 0 and eval_train:
            agent.eval()
            score_summary = {}
            for key in eval_train_env:
                agent.env = eval_train_env[key]
                agent.test(iters=None, feedback="argmax")
                scores, __ = train_evaluator[key].score(agent.get_results())
                score_summary[key] = scores

                writer.add_scalar(f"eval_{key}/traj_lengths", scores['lengths'], ep)
                writer.add_scalar(f"eval_{key}/traj_steps", scores['steps'], ep)
                writer.add_scalar(f"eval_{key}/nav_error", scores['nav_error'], ep) 
                writer.add_scalar(f"eval_{key}/oracle_error", scores['oracle_error'], ep)
                writer.add_scalar(f"eval_{key}/success_rate", scores['success_rate'], ep)
                writer.add_scalar(f"eval_{key}/oracle_rate", scores['oracle_rate'], ep)
                writer.add_scalar(f"eval_{key}/spl", scores['spl'], ep)
                writer.add_scalar(f"eval_{key}/ndtw", scores['ndtw'], ep)
                writer.add_scalar(f"eval_{key}/sdtw", scores['sdtw'], ep)
            
            utils.prettyprint(score_summary)

        if ep % train_cfg.EVAL_INTERVAL == 0:
            agent.eval()

            score_summary = {}
            # eval validation env
            for key in valid_env:
                agent.env = valid_env[key]
                agent.test(iters=None, feedback="argmax")
                scores, __ = valid_evaluator[key].score(agent.get_results())
                score_summary[key] = scores

                writer.add_scalar(f"{key}/traj_lengths", scores['lengths'], ep)
                writer.add_scalar(f"{key}/traj_steps", scores['steps'], ep)
                writer.add_scalar(f"{key}/nav_error", scores['nav_error'], ep) 
                writer.add_scalar(f"{key}/oracle_error", scores['oracle_error'], ep)
                writer.add_scalar(f"{key}/success_rate", scores['success_rate'], ep)
                writer.add_scalar(f"{key}/oracle_rate", scores['oracle_rate'], ep)
                writer.add_scalar(f"{key}/spl", scores['spl'], ep)
                writer.add_scalar(f"{key}/ndtw", scores['ndtw'], ep)
                writer.add_scalar(f"{key}/sdtw", scores['sdtw'], ep)
                
                if key in best_val:
                    if scores['success_rate'] > best_val[key]['success_rate']:
                        best_val[key]['success_rate'] = scores['success_rate']
                        # cur_save_path = osp.join(cfg.OUTPUT.CKPT_DIR, f"best_{key}.pt")
                        cur_save_path = osp.join(output_ckpt_dir, "best_{}_SR:{:.4f}.pt".format(key, scores['success_rate']))
                        clean_dir(output_ckpt_dir, clean_key=f"best_{key}")
                        agent.save_model(cur_save_path, cfg=cfg, last_epoch=ep)
                        print(f"\t Saving best {key} model with SR={best_val[key]['success_rate']:.4f} into {cur_save_path}.")
                        logger.info(f"\t Saving best {key} model with SR={best_val[key]['success_rate']:.4f} into {cur_save_path}.")

            utils.prettyprint(score_summary)

        clean_dir(output_ckpt_dir, clean_key=f"latest_ep")
        agent.save_model(osp.join(output_ckpt_dir, f"latest_ep{ep}.pt"), cfg=cfg, last_epoch=ep)
        last_time = time.time()
    
    return agent


###############################
#      Self-Monitoring        #
###############################

def train_selfmonitor(cfg, agent:BasicR2RAgent, tsboard_dir:str, train_env:R2RBatch, valid_env:dict,
    eval_train:bool=False, eval_train_env:dict=None,
):
    ''' train the agent '''
    time_str = time.strftime('%Y-%m%d-%H:%M', time.localtime(time.time()))
    summary_save_dir = os.path.join(tsboard_dir, time_str)
    writer = SummaryWriter(logdir=summary_save_dir)
    print(f"\t You can find tensorboard summary at {summary_save_dir}.")
    logger.info(f"\t You can find tensorboard summary at {summary_save_dir}.")

    train_cfg = cfg.TRAIN

    if cfg.OUTPUT.RESUME != "":
        print(f"\t LOAD the {cfg.MODEL.NAME} model from {cfg.OUTPUT.RESUME} ...")
        logger.info(f"\t LOAD the {cfg.MODEL.NAME} model from {cfg.OUTPUT.RESUME} ...")
        ckpt = agent.load_model(osp.join(cfg.OUTPUT.CKPT_DIR, f"{cfg.OUTPUT.RESUME}.pt"), cuda=cfg.TRAIN.DEVICE)
        if "train_cfg" in ckpt: train_cfg = ckpt["train_cfg"]
        if "last_epoch" in ckpt: train_cfg.START_EPOCH = ckpt["last_epoch"] + 1

    which_optim = optim_switcher.get(train_cfg.OPTIM, torch.optim.Adam)
    trainable_params = filter(lambda p: p.requires_grad, \
        list(agent.encoder.parameters())+list(agent.decoder.parameters()))
    optimizer = which_optim(trainable_params, lr=train_cfg.LR)

    if cfg.DATA.NAME in ["R2R", "CLR2R", "FGR2R"]:
        valid_evaluator = {
            "val_unseen": Evaluation(splits=['val_unseen']), 
            "val_seen": Evaluation(splits=['val_seen'])
        }
    elif cfg.DATA.NAME == "RxR":
        valid_evaluator = {
            "val_unseen": Evaluation(splits=['val_unseen'], data_name="RxR", data_dir=cfg.DATA.DATA_DIR), 
            "val_seen": Evaluation(splits=['val_seen'], data_name="RxR", data_dir=cfg.DATA.DATA_DIR)
        }
    elif cfg.DATA.NAME == "Mixed":
        valid_evaluator = {
            "r2r_val_unseen": Evaluation(splits=['val_unseen']), 
            "r2r_val_seen": Evaluation(splits=['val_seen']),
            "rxr_val_unseen": Evaluation(splits=['val_unseen'], data_name="RxR", data_dir="tasks/R2R-judy/data/RxR-en"), 
            "rxr_val_seen": Evaluation(splits=['val_seen'], data_name="RxR", data_dir="tasks/R2R-judy/data/RxR-en")
        }
    else: raise NotImplementedError
    best_val = {key: {"success_rate": 0.} for key in valid_evaluator.keys()}
    
    if eval_train:
        train_evaluator = {}
        for k in range(1, 6):
            train_evaluator[f"round_{k}"] = Evaluation(
                [f'train_round[{k}]_v3'], data_name="CLR2R", data_dir="tasks/R2R-judy/data/CLR2Rv3")
        assert eval_train_env is not None, "Please give the eval environment!"
        assert eval_train_env.keys() == train_evaluator.keys(), "Key not Match!"

    output_ckpt_dir = os.path.join(cfg.OUTPUT.CKPT_DIR, time_str)
    if not os.path.exists(output_ckpt_dir): os.makedirs(output_ckpt_dir)
    print(f"\t You can find checkpoints at {output_ckpt_dir}.")
    logger.info(f"\t You can find checkpoints at {output_ckpt_dir}.")

    start_time, last_time = time.time(), time.time()

    for ep in range(train_cfg.START_EPOCH, train_cfg.MAX_EPOCH+1):

        agent.env = train_env
        agent.train()
        agent.reset_loss()
        
        for __ in range(1, train_cfg.ITER_PER_EPOCH+1):
            agent.rollout(train_ml=True, feedback=cfg.AGENT.FEEDBACK, lamb=train_cfg.PROGMONITOR_WEIGHT)

            optimizer.zero_grad()
            agent.ml_loss.backward()
            optimizer.step()
        
        # recoding training infomation
        epoch_loss = sum(agent.losses)
        avg_iter_loss = epoch_loss / len(agent.losses)
        max_iter_loss = max(agent.losses)
        min_iter_loss = min(agent.losses)
        writer.add_scalar("train/ml_epoch", epoch_loss, ep)         # machine learning loss
        writer.add_scalar("train/ml_iter_avg", avg_iter_loss, ep)   # = action selection loss + progress loss
        writer.add_scalar("train/ml_iter_max", max_iter_loss, ep)
        writer.add_scalar("train/ml_iter_min", min_iter_loss, ep)
        writer.add_scalar("train/progress_loss", sum(agent.progress_losses), ep)

        # routine print
        epoch_time_cost = (time.time()-last_time)/60
        remaining_time_cost = ((time.time()-start_time)/(60 * (ep+1-train_cfg.START_EPOCH))) * (train_cfg.MAX_EPOCH - ep)
        train_print_str = f"\t Epoch [{ep}/{train_cfg.MAX_EPOCH}], Time Cost {epoch_time_cost:.2f}min/ep," + \
                          f" Remaining {remaining_time_cost:.2f} min, Train Loss: {epoch_loss:.4f}," + \
                          f" Iter [AVG: {avg_iter_loss:.4f}, MIN: {min_iter_loss:.4f}, MAX: {max_iter_loss:.4f}]," + \
                          f" Progress Loss: {sum(agent.progress_losses):.4f} ."
        print(train_print_str)
        logger.info(train_print_str)

        # evaluation
        if ep % train_cfg.EVAL_INTERVAL == 0 and eval_train:
            agent.eval()
            score_summary = {}
            for key in eval_train_env:
                agent.env = eval_train_env[key]
                agent.test(iters=None, feedback="argmax")
                scores, __ = train_evaluator[key].score(agent.get_results())
                score_summary[key] = scores

                writer.add_scalar(f"eval_{key}/traj_lengths", scores['lengths'], ep)
                writer.add_scalar(f"eval_{key}/traj_steps", scores['steps'], ep)
                writer.add_scalar(f"eval_{key}/nav_error", scores['nav_error'], ep) 
                writer.add_scalar(f"eval_{key}/oracle_error", scores['oracle_error'], ep)
                writer.add_scalar(f"eval_{key}/success_rate", scores['success_rate'], ep)
                writer.add_scalar(f"eval_{key}/oracle_rate", scores['oracle_rate'], ep)
                writer.add_scalar(f"eval_{key}/spl", scores['spl'], ep)
                writer.add_scalar(f"eval_{key}/ndtw", scores['ndtw'], ep)
                writer.add_scalar(f"eval_{key}/sdtw", scores['sdtw'], ep)
            
            utils.prettyprint(score_summary)

        if ep % train_cfg.EVAL_INTERVAL == 0:
            agent.eval()

            score_summary = {}
            # eval validation env
            for key in valid_env:
                agent.env = valid_env[key]
                agent.test(iters=None, feedback="argmax")
                scores, __ = valid_evaluator[key].score(agent.get_results())
                score_summary[key] = scores

                writer.add_scalar(f"{key}/traj_lengths", scores['lengths'], ep)
                writer.add_scalar(f"{key}/traj_steps", scores['steps'], ep)
                writer.add_scalar(f"{key}/nav_error", scores['nav_error'], ep) 
                writer.add_scalar(f"{key}/oracle_error", scores['oracle_error'], ep)
                writer.add_scalar(f"{key}/success_rate", scores['success_rate'], ep)
                writer.add_scalar(f"{key}/oracle_rate", scores['oracle_rate'], ep)
                writer.add_scalar(f"{key}/spl", scores['spl'], ep)
                writer.add_scalar(f"{key}/ndtw", scores['ndtw'], ep)
                writer.add_scalar(f"{key}/sdtw", scores['sdtw'], ep)
                
                if key in best_val:
                    if scores['success_rate'] > best_val[key]['success_rate']:
                        best_val[key]['success_rate'] = scores['success_rate']
                        cur_save_path = osp.join(output_ckpt_dir, "best_{}_SR:{:.4f}.pt".format(key, scores['success_rate']))
                        clean_dir(output_ckpt_dir, clean_key=f"best_{key}")
                        agent.save_model(cur_save_path, cfg=cfg, last_epoch=ep)
                        print(f"\t Saving best {key} model with SR={best_val[key]['success_rate']:.4f} into {cur_save_path}.")
                        logger.info(f"\t Saving best {key} model with SR={best_val[key]['success_rate']:.4f} into {cur_save_path}.")

            utils.prettyprint(score_summary)

        clean_dir(output_ckpt_dir, clean_key="latest_avgloss")
        agent.save_model(osp.join(
            output_ckpt_dir, "latest_avgloss:{:.4f}_ep_{}.pt".format(avg_iter_loss, ep)), 
            cfg=cfg, last_epoch=ep)
        # agent.save_model(osp.join(cfg.OUTPUT.CKPT_DIR, "latest.pt"), cfg=cfg, last_epoch=ep)
        last_time = time.time()
    
    return agent


###############################
#    Environmental Dropout    #
###############################

def train_envdrop(cfg, agent:BasicR2RAgent, tsboard_dir:str, train_env:R2RBatch, valid_env:dict,
    eval_train:bool=False, eval_train_env:dict=None,
):
    ''' train the agent '''
    summary_save_dir = os.path.join(tsboard_dir, 
        time.strftime('%Y-%m%d-%H:%M', time.localtime(time.time())))
    writer = SummaryWriter(logdir=summary_save_dir)
    print(f"\t You can find tensorboard summary at {summary_save_dir}.")
    logger.info(f"\t You can find tensorboard summary at {summary_save_dir}.")

    train_cfg = cfg.TRAIN

    if cfg.OUTPUT.RESUME != "":
        print(f"\t LOAD the {cfg.MODEL.NAME} model from {cfg.OUTPUT.RESUME} ...")
        logger.info(f"\t LOAD the {cfg.MODEL.NAME} model from {cfg.OUTPUT.RESUME} ...")
        ckpt = agent.load_model(osp.join(cfg.OUTPUT.CKPT_DIR, f"{cfg.OUTPUT.RESUME}.pt"))
        if "train_cfg" in ckpt: train_cfg = ckpt["train_cfg"]
        if "last_epoch" in ckpt: train_cfg.START_EPOCH = ckpt["last_epoch"] + 1
    
    which_optim = optim_switcher.get(train_cfg.OPTIM, torch.optim.Adam)
    optimizer = which_optim(agent.trainable_params(), lr=train_cfg.LR)

    if cfg.DATA.NAME != "RxR":
        valid_evaluator = {
            "val_unseen": Evaluation(splits=['val_unseen']), 
            "val_seen":   Evaluation(splits=['val_seen']),
        }
    else:
        valid_evaluator = {
            "val_unseen": Evaluation(splits=['val_unseen'], data_name=cfg.DATA.NAME, data_dir=cfg.DATA.DATA_DIR), 
            "val_seen":   Evaluation(splits=['val_seen'], data_name=cfg.DATA.NAME, data_dir=cfg.DATA.DATA_DIR),
        }
    best_val = {'val_seen': {"success_rate": 0.}, 'val_unseen': {"success_rate": 0.}}

    if eval_train:
        train_evaluator = {}
        for k in range(1, 6):
            train_evaluator[f"round_{k}"] = Evaluation(
                [f'train_round[{k}]_v3'], data_name="CLR2R", data_dir="tasks/R2R-judy/data/CLR2Rv3")
        assert eval_train_env is not None, "Please give the eval environment!"
        assert eval_train_env.keys() == train_evaluator.keys(), "Key not Match!"

    start_time, last_time = time.time(), time.time()

    for ep in range(train_cfg.START_EPOCH, train_cfg.MAX_EPOCH+1):

        agent.env = train_env
        agent.train()
        agent.reset_loss()
        
        for __ in range(1, train_cfg.ITER_PER_EPOCH+1):
            if cfg.AGENT.FEEDBACK == "sample":
                agent.rollout(train_ml=True, train_rl=False, feedback="teacher")
                ml_loss = agent.loss["ml_loss"]
                agent.rollout(train_ml=False, train_rl=True, restart=True, feedback="sample")
                rl_loss = agent.loss["rl_loss"]
            elif cfg.AGENT.FEEDBACK == "teacher":
                agent.rollout(train_ml=True, train_rl=False, feedback="teacher")
                ml_loss = agent.loss["ml_loss"]
                rl_loss = 0.0
            cur_loss = ml_loss + rl_loss

            optimizer.zero_grad()
            cur_loss.backward()
            torch.nn.utils.clip_grad_norm(agent.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(agent.decoder.parameters(), 40.)
            optimizer.step()

            agent.losses.append(cur_loss.item())
        
        # recoding training infomation
        epoch_loss = sum(agent.losses)
        avg_iter_loss = epoch_loss / len(agent.losses)
        max_iter_loss = max(agent.losses)
        min_iter_loss = min(agent.losses)
        writer.add_scalar("train/ml+rl_epoch", epoch_loss, ep)          # machine learning loss
        writer.add_scalar("train/ml+rl_iter_avg", avg_iter_loss, ep)    # = action selection loss
        writer.add_scalar("train/ml+rl_iter_max", max_iter_loss, ep)
        writer.add_scalar("train/ml+rl_iter_min", min_iter_loss, ep)
        total = max(sum(agent.logs['total']), 1)
        length = max(len(agent.logs['critic_loss']), 1)
        critic_loss = sum(agent.logs['critic_loss']) / total          #/ length / args.batchSize
        entropy = sum(agent.logs['entropy']) / total                  #/ length / args.batchSize
        writer.add_scalar("train/critic_loss", critic_loss, ep)
        writer.add_scalar("train/policy_entropy", entropy, ep)
        writer.add_scalar("train/total_actions", total, ep)
        writer.add_scalar("train/max_length", length, ep)

        # routine print
        epoch_time_cost = (time.time()-last_time)/60
        remaining_time_cost = ((time.time()-start_time)/(60 * (ep+1-train_cfg.START_EPOCH))) * (train_cfg.MAX_EPOCH - ep)
        train_print_str = f"\t Epoch [{ep}/{train_cfg.MAX_EPOCH}], Time Cost {epoch_time_cost:.2f}min/ep," + \
                          f" Remaining {remaining_time_cost:.2f} min, Train Loss: {epoch_loss:.4f}," + \
                          f" Iter [AVG: {avg_iter_loss:.4f}, MIN: {min_iter_loss:.4f}, MAX: {max_iter_loss:.4f}]." + \
                          f" Total actions: {total}, Max length: {length}."
        print(train_print_str)
        logger.info(train_print_str)

        # evaluation
        if ep % train_cfg.EVAL_INTERVAL == 0 and eval_train:
            agent.eval()
            score_summary = {}
            for key in eval_train_env:
                agent.env = eval_train_env[key]
                agent.test(iters=None, feedback="argmax")
                scores, __ = valid_evaluator[key].score(agent.get_results())
                score_summary[key] = scores

                writer.add_scalar(f"eval_{key}/traj_lengths", scores['lengths'], ep)
                writer.add_scalar(f"eval_{key}/traj_steps", scores['steps'], ep)
                writer.add_scalar(f"eval_{key}/nav_error", scores['nav_error'], ep) 
                writer.add_scalar(f"eval_{key}/oracle_error", scores['oracle_error'], ep)
                writer.add_scalar(f"eval_{key}/success_rate", scores['success_rate'], ep)
                writer.add_scalar(f"eval_{key}/oracle_rate", scores['oracle_rate'], ep)
                writer.add_scalar(f"eval_{key}/spl", scores['spl'], ep)
                writer.add_scalar(f"eval_{key}/ndtw", scores['ndtw'], ep)
                writer.add_scalar(f"eval_{key}/sdtw", scores['sdtw'], ep)
            
            utils.prettyprint(score_summary)

        if ep % train_cfg.EVAL_INTERVAL == 0:
            agent.eval()

            score_summary = {}
            # eval validation env
            for key in valid_env:
                agent.env = valid_env[key]
                agent.test(iters=None, feedback="argmax")
                scores, __ = valid_evaluator[key].score(agent.get_results())
                score_summary[key] = scores

                writer.add_scalar(f"{key}/traj_lengths", scores['lengths'], ep)
                writer.add_scalar(f"{key}/traj_steps", scores['steps'], ep)
                writer.add_scalar(f"{key}/nav_error", scores['nav_error'], ep) 
                writer.add_scalar(f"{key}/oracle_error", scores['oracle_error'], ep)
                writer.add_scalar(f"{key}/success_rate", scores['success_rate'], ep)
                writer.add_scalar(f"{key}/oracle_rate", scores['oracle_rate'], ep)
                writer.add_scalar(f"{key}/spl", scores['spl'], ep)
                writer.add_scalar(f"{key}/ndtw", scores['ndtw'], ep)
                writer.add_scalar(f"{key}/sdtw", scores['sdtw'], ep)
                
                if key in best_val:
                    if scores['success_rate'] > best_val[key]['success_rate']:
                        best_val[key]['success_rate'] = scores['success_rate']
                        cur_save_path = osp.join(cfg.OUTPUT.CKPT_DIR, "best_{}_SR:{:.4f}.pt".format(key, scores['success_rate']))
                        clean_dir(cfg.OUTPUT.CKPT_DIR, clean_key=f"best_{key}")
                        agent.save_model(cur_save_path, cfg=cfg, last_epoch=ep)
                        print(f"\t Saving best {key} model with SR={best_val[key]['success_rate']:.4f} into {cur_save_path}.")
                        logger.info(f"\t Saving best {key} model with SR={best_val[key]['success_rate']:.4f} into {cur_save_path}.")

            utils.prettyprint(score_summary)

        clean_dir(cfg.OUTPUT.CKPT_DIR, clean_key="latest_avgloss")
        agent.save_model(osp.join(cfg.OUTPUT.CKPT_DIR, "latest_avgloss:{:.4f}.pt".format(avg_iter_loss)), cfg=cfg, last_epoch=ep)
        # agent.save_model(osp.join(cfg.OUTPUT.CKPT_DIR, "latest.pt"), cfg=cfg, last_epoch=ep)
        last_time = time.time()
    
    return agent   



def clean_dir(save_dir, clean_key):
    file_names = os.listdir(save_dir)
    for fn in file_names:
        if clean_key in fn:
            os.remove(os.path.join(save_dir, fn))
    
