import os
import os.path as osp
import time
import logging
logger = logging.getLogger("main.curriculum")

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class NaiveCurriculum(object):

    def __init__(self, switch_epoch:int=20, reverse=False):
        self.switch_epoch = switch_epoch
        self.reverse = reverse

    def train(self, cfg, agent:BasicR2RAgent, tsboard_dir:str, train_env:R2RBatch, valid_env:dict):
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
        optimizer = which_optim(agent.trainable_params(), lr=train_cfg.LR)

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
        start_time, last_time = time.time(), time.time()

        output_ckpt_dir = os.path.join(cfg.OUTPUT.CKPT_DIR, time_str)
        if not os.path.exists(output_ckpt_dir): os.makedirs(output_ckpt_dir)
        print(f"\t You can find checkpoints at {output_ckpt_dir}.")
        logger.info(f"\t You can find checkpoints at {output_ckpt_dir}.")

        for ep in range(train_cfg.START_EPOCH, train_cfg.MAX_EPOCH+1):
            # naive curriculum will change the task selection
            agent.env = self.curriculum_strategy(train_env, ep)
            agent.train()
            agent.reset_loss()

            # naive curriculum will not change batch sample strategy
            for __ in range(1, train_cfg.ITER_PER_EPOCH+1):
                if cfg.MODEL.NAME == "ENVDROP":
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
                else:
                    agent.rollout(train_ml=True, feedback=cfg.AGENT.FEEDBACK)

                    optimizer.zero_grad()
                    agent.ml_loss.backward()
                    optimizer.step()
            
            # recoding training infomation
            epoch_loss = sum(agent.losses)
            avg_iter_loss = epoch_loss / len(agent.losses)
            max_iter_loss = max(agent.losses)
            min_iter_loss = min(agent.losses)
            writer.add_scalar("train/ml_epoch", epoch_loss, ep)
            writer.add_scalar("train/ml_iter_avg", avg_iter_loss, ep)
            writer.add_scalar("train/ml_iter_max", max_iter_loss, ep)
            writer.add_scalar("train/ml_iter_min", min_iter_loss, ep)
            if cfg.MODEL.NAME == "SELF-MONITOR":
                writer.add_scalar("train/progress_epoch", sum(agent.progress_losses), ep)
            if cfg.MODEL.NAME == "ENVDROP":
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
                              f" Iter [AVG: {avg_iter_loss:.4f}, MIN: {min_iter_loss:.4f}, MAX: {max_iter_loss:.4f}]."
            if cfg.MODEL.NAME == "SELF-MONITOR":
                train_print_str = train_print_str + f" Progress Loss: {sum(agent.progress_losses):.4f} ."
            if cfg.MODEL.NAME == "ENVDROP":
                train_print_str = train_print_str + f" Total actions: {total}, Max length: {length}."
            print(train_print_str)
            logger.info(train_print_str)
            
            # evaluation
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
            agent.save_model(osp.join(output_ckpt_dir, "latest_avgloss:{:.4f}.pt".format(avg_iter_loss)), cfg=cfg, last_epoch=ep)
            last_time = time.time()

        return agent  

    def curriculum_strategy(self, train_env, cur_epoch):
        idx = 1 + (cur_epoch-1) // self.switch_epoch
        if idx <= 4: return train_env[f'round_{idx}']
        else: return train_env['round_5']



class SelfPacedCurriculum(object):
    """ Jiang, L., Meng, D., Zhao, Q., Shan, S., & Hauptmann, A. 
        (2015). Self-Paced Curriculum Learning. AAAI. """

    def __init__(self, train_env, device, pace_func:str="linear", 
        init_lamb:float=0.1, init_weight_ctrl:float=0.5,
        miu:float=0.1, interval:int=5, strategy:str="epoch", burn_in:int=10,
    ):
        """ Init the params.
        
        lambda:             the age of model, a aprameter for controlling the learning pace
        weight:             reflecting the sample importance
        stepsize:           update lambda if it is too small
        update_interval:    when to update lambda
        update_strategy:    how to update lambda, batch | epoch
        """
        self.train_env = train_env
        self.device = device
        self.pace_func = pace_func
        self.dim = len(train_env)
        self.a = torch.from_numpy(train_env.a).to(device)
        self.c = torch.tensor(train_env.c, device=device)

        self.lamb = torch.tensor(init_lamb, device=device)
        self.weight = self._init_weight_(init_weight_ctrl)

        self.stepsize = miu
        self.burn_in = burn_in
        self.update_interval = interval
        self.update_strategy = strategy
    
    def _init_weight_(self, val):
        weight = torch.ones(self.dim) * val
        for item in self.train_env.data:
            item_idx = self.train_env.index(item)
            item_diff = self.a[item_idx]
            if item_diff <= 2: weight[item_idx] = 1.0
        return weight.to(self.device)

    def train(self, cfg, agent:BasicR2RAgent, tsboard_dir:str, train_env, valid_env:dict, 
        eval_train:bool=False, eval_train_env:dict=None, **kwargs,
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
        optimizer = which_optim(agent.trainable_params(), lr=train_cfg.LR)

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

        start_time, last_time = time.time(), time.time()

        output_ckpt_dir = os.path.join(cfg.OUTPUT.CKPT_DIR, time_str)
        if not os.path.exists(output_ckpt_dir): os.makedirs(output_ckpt_dir)
        print(f"\t You can find checkpoints at {output_ckpt_dir}.")
        logger.info(f"\t You can find checkpoints at {output_ckpt_dir}.")

        # record the most recent loss for each item
        loss_for_item = torch.zeros(self.dim, device=self.device)

        for ep in range(train_cfg.START_EPOCH, train_cfg.MAX_EPOCH+1):
            # self-paced curriculum will not change the task selection
            # it just sample as normal
            # the control is made at self.weight and self.lamb
            agent.env = self.train_env
            agent.train()
            agent.reset_loss()
            record_losses = []

            for __ in range(1, train_cfg.ITER_PER_EPOCH+1):
                if cfg.MODEL.NAME == "ENVDROP":
                    if cfg.AGENT.FEEDBACK == "sample":
                        agent.rollout(train_ml=True, train_rl=False, train_cl=True, feedback="teacher")
                        ml_loss = agent.loss["ml_loss"]
                        agent.rollout(train_ml=False, train_rl=True, train_cl=True, restart=True, feedback="sample")
                        rl_loss = agent.loss["rl_loss"]
                    elif cfg.AGENT.FEEDBACK == "teacher":
                        agent.rollout(train_ml=True, train_rl=False, train_cl=True, feedback="teacher")
                        ml_loss = agent.loss["ml_loss"]
                        rl_loss = 0.0
                    cur_loss = ml_loss + rl_loss
                    cur_batch_idx = agent.env.cur_batch_index
                    batch_loss = torch.dot(self.weight[cur_batch_idx], cur_loss)
                else:
                    agent.rollout(train_ml=True, train_cl=True, feedback=cfg.AGENT.FEEDBACK) 
                    cur_loss = agent.ml_loss          
                    cur_batch_idx = agent.env.cur_batch_index
                    batch_loss = torch.dot(self.weight[cur_batch_idx], cur_loss) / self.weight[cur_batch_idx].sum()

                optimizer.zero_grad()
                batch_loss.backward()
                if cfg.MODEL.NAME == "ENVDROP":
                    torch.nn.utils.clip_grad_norm(agent.encoder.parameters(), 40.)
                    torch.nn.utils.clip_grad_norm(agent.decoder.parameters(), 40.)
                optimizer.step()
                # recoding
                record_losses.append(batch_loss.item())
                with torch.no_grad(): 
                    if cfg.MODEL.NAME == "ENVDROP": 
                        loss_for_item[cur_batch_idx] = ml_loss * len(cur_batch_idx)
                    else: loss_for_item[cur_batch_idx] = cur_loss
            
            # recoding training infomation
            epoch_loss = sum(record_losses)
            avg_iter_loss = epoch_loss / len(record_losses)
            max_iter_loss = max(record_losses)
            min_iter_loss = min(record_losses)
            writer.add_scalar("train/ml_epoch", epoch_loss, ep)
            writer.add_scalar("train/ml_iter_avg", avg_iter_loss, ep)
            writer.add_scalar("train/ml_iter_max", max_iter_loss, ep)
            writer.add_scalar("train/ml_iter_min", min_iter_loss, ep)
            if cfg.MODEL.NAME == "SELF-MONITOR":
                writer.add_scalar("train/progress_epoch", sum(agent.progress_losses), ep)
            if cfg.MODEL.NAME == "ENVDROP":
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
                              f" Iter [AVG: {avg_iter_loss:.4f}, MIN: {min_iter_loss:.4f}, MAX: {max_iter_loss:.4f}]."
            if cfg.MODEL.NAME == "SELF-MONITOR":
                train_print_str = train_print_str + f" Progress Loss: {sum(agent.progress_losses):.4f} ."
            if cfg.MODEL.NAME == "ENVDROP":
                train_print_str = train_print_str + f" Total actions: {total}, Max length: {length}."
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

            # TODO: update parameter
            if ep >= self.burn_in and ep % self.update_interval == 0:
                loss_np = loss_for_item.detach().cpu().numpy()
                q_25, q_50, q_75 = np.percentile(loss_np, 25), np.percentile(loss_np, 50), np.percentile(loss_np, 75)
                if self.lamb < loss_for_item.max().item(): self.lamb += self.stepsize
                else: self.lamb += self.stepsize / 2
                loss_distribution_str = f"\t Now lambda={self.lamb}, whereas [0, 25, 50, 75, 100]% percentile of sample loss is "+\
                    "[{:.4f}, {:4f}, {:.4f}, {:4f}, {:.4f}]".format(loss_np.min(), q_25, q_50, q_75, loss_np.max())
                print(loss_distribution_str)
                logger.info(loss_distribution_str)
                self.update_weight(loss_for_item)
                
                writer.add_histogram("sample_weight", self.weight.detach().cpu().numpy(), ep)
                writer.add_histogram("sample_loss", loss_np, ep)

            # clean_dir(output_ckpt_dir, clean_key="latest_avgloss")
            agent.save_model(osp.join(output_ckpt_dir, "latest_avgloss:{:.4f}_ep_{}.pt".format(avg_iter_loss, ep)), cfg=cfg, last_epoch=ep)
            last_time = time.time()
        
        return agent

    def update_weight(self, loss:np.ndarray):
        if self.update_strategy == "epoch":
            self._update_epoch_(loss)
        else: raise NotImplementedError
    
    def _update_epoch_(self, epoch_loss):
        zeta = 1 - self.lamb
        mask = (epoch_loss >= self.lamb)
        self.weight[mask] = 0.01
        if self.pace_func == "log":
            self.weight[~mask] = torch.log(epoch_loss[~mask] + zeta) / torch.log(zeta)
        elif self.pace_func == "linear":
            self.weight[~mask] = 1 - epoch_loss[~mask] / self.lamb
        elif self.pace_func == "binary":
            self.weight[~mask] = 1.0
        else: raise NotImplementedError
        # aviod zero weight
        self.weight[self.weight < 0.01] = 0.01
        # project to the curriculum region
        if torch.dot(self.a, self.weight) > self.c:
            a_norm = torch.norm(self.a, p=2)
            new_weight = self.weight + self.a * \
                (self.c - torch.dot(self.a, self.weight)) / (a_norm*a_norm)
            # aviod negative weight
            new_weight[new_weight <= 0.0] = 0.001
            self.weight = new_weight
        
        info_str="\t Sample weight:"
        for k in range(1, 6):
            wk = self.weight[self.a == k]
            info_str += "\n\t\t Round[{}], avg: {:.3f}, min: {:.3f}, max: {:.3f}".format(
                k, wk.mean().item(), wk.min().item(), wk.max().item())
        print(info_str)
        logger.info(info_str)



def clean_dir(save_dir, clean_key):
    file_names = os.listdir(save_dir)
    for fn in file_names:
        if clean_key in fn:
            os.remove(os.path.join(save_dir, fn))

