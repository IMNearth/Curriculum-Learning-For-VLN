import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import src.utils as utils
import src.environ as environ
from .base import BasicR2RAgent
import src.model as M


class FollowerAgent(BasicR2RAgent):
    """ Follower model. 
        Speaker-Follower Models for Vision-and-Language Navigation. 
        Fried et al. NeurIPS(2018) """

    def __init__(self, 
        model_cfg, 
        results_dir:str, 
        device:int,
        env:environ.R2RBatch, 
        tokenizer:utils.Tokenizer, 
        glove:np.ndarray=None,
        episode_len:int=20,
    ):
        super(FollowerAgent, self).__init__(
            results_dir, device, env, tokenizer, episode_len=episode_len)
        
        self.cfg = model_cfg
        self.action_emb_size = self.feature_size
        
        # Init the encoder and decoder
        self.encoder = M.EncoderLSTM(
            self.tokenizer.vocab_size(), 
            self.cfg.WORD_EMB_SIZE, 
            self.cfg.HIDDEN_SIZE, 
            padding_idx=utils.pad_idx, 
            drop_ratio=self.cfg.DROP_RATE, 
            bidirectional=self.cfg.ENC_BIDIRECTION, 
            num_layers=self.cfg.ENC_LAYERS, 
            glove=glove
        )
        self.decoder = M.AttnDecoderLSTM(
            self.cfg.HIDDEN_SIZE, 
            drop_ratio=self.cfg.DROP_RATE,
            action_embed_size=self.action_emb_size, 
            feature_size=self.feature_size
        )
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_id)
        self.curriculum_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_id, reduction="none")
    
    def rollout(self, 
        train_ml=True, train_rl=False, train_cl=False, reset=True, restart=False,
        speaker=None, avoid_cyclic=False, feedback="sample"
    ):
        """
        :param train_ml:        The weight to train with maximum likelihood
        :param train_rl:        whether to use RL in training
        :param reset:           Reset the environment
        :param speaker:         Speaker used in back translation.
                                If the speaker is not None, use back translation.
                                O.w., normal training
        :param avoid_cyclic:    whether to mask visited viewpoints
        """
        obs = self.env.reset(restart=restart) if reset else self.env.observe()
        batch_size = len(obs)

        # 1. Get the instruction representation
        seq_tensor, seq_mask, seq_length = self._instr_variable(obs)
        #    and encode the sequence
        ctx, h_t, c_t = self.encoder(seq_tensor, seq_length)

        # 2. Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpointId'], ob['heading'], ob['elevation'])]
        } for ob in obs]

        # 3. Initialize actions and the tracking state
        a_t_prev = Variable(torch.zeros(\
            batch_size, self.action_emb_size, device=self.device),\
            requires_grad=False)
        
        # 4. Initialize reward shaping
        last_dist = np.array([ob['distance'] for ob in obs], np.float32)
        rewards = []

        # 5. Initialize other recoding variables
        ended = np.array([False] * batch_size)
        visited = [set() for _ in range(batch_size)]
        self.ml_loss = 0. if not train_cl else torch.zeros(batch_size, device=self.device)

        # 6. Start rolling out
        for t in range(self.episode_len):
            img_feature = self._feature_variable(obs)  # (batch, num_views, feature_size)
            a_t_cands, cands_lengs = self._candidate_variable(obs)

            # (1) Run the decoder
            logits, (h_t, c_t), (alpha_c, alpha_v) = self.decoder(
                img_feature, a_t_prev, a_t_cands, h_t, c_t, ctx, seq_mask)
            
            # (2) Mask outputs where agent can't move forward
            candidate_mask = utils.length2mask(cands_lengs, self.device)
            if avoid_cyclic:
                for i, ob in enumerate(obs):
                    visited[i].add(ob['viewpointId'])
                    for j, c in ob['candidates']:
                        if c['viewpointId'] in visited[i]:
                            candidate_mask[i][j] = 1
            logits.masked_fill_(candidate_mask, -float('inf'))

            # (3) Get the ground truth action and Compute the loss
            target = self._teacher_action(obs, ended)
            if not train_cl: self.ml_loss += self.criterion(logits, target)
            else: self.ml_loss += self.curriculum_criterion(logits, target)

            # (4) Determine the action!
            if feedback == "teacher":  # teacher-forcing
                a_t = target
            elif feedback == "argmax": # student-forcing
                _, a_t = logits.max(1)
            elif feedback == "sample": # sample an action from model
                probs = F.softmax(logits, 1)
                c = torch.distributions.Categorical(probs)
                a_t = c.sample()
            else: raise NotImplementedError
            
            cpu_a_t = a_t.detach().cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                # The last action is <stop> / Now at target point / already ended
                # Change the <stop> and ignore action to -1, which means the agent will not move
                if next_id == (len(obs[i]['candidates'])) or next_id == self.ignore_id or ended[i]:
                    cpu_a_t[i] = -1

            # (5) Make action and get the new state
            #     also trajectories are recorded ...
            obs = self.move_and_observe(cpu_a_t, obs, traj)   

            # (6) Calculate the mask and reward
            dist = np.array([ob['distance'] for ob in obs], np.float32)
            is_stop = (cpu_a_t == -1)
            reward = (
                is_stop * np.sign(last_dist - dist) + \
                (1 - is_stop) * (2*(dist < 3)-1) * 2
            ) * (1-ended)

            rewards.append(reward)
            last_dist[:] = dist

            # (7) Update the previous action
            a_t_prev = a_t_cands[np.arange(batch_size), np.maximum(cpu_a_t, 0), :].detach()
            
            # (8) Update the finished actions
            ended[:] = np.logical_or(ended, is_stop)
            if ended.all(): break

        # 7. Record the loss
        if not train_cl and not restart: self.losses.append(self.ml_loss.item())

        return traj

    def running_state(self, h_t, c_t, a_t_prev, **kwargs):
        return (h_t, c_t, a_t_prev)

    def decode_obervation(self, obs, h_t, c_t, a_t_prev, ctx, ctx_mask, ended, **kwargs):
        """ Agent dependent decoding process. """
        batch_size = len(obs)

        img_feature = self._feature_variable(obs)  # (batch, num_views, feature_size)
        a_t_cands, cands_lengs = self._candidate_variable(obs)
        
        logits, (h_t, c_t), __ = self.decoder(
            img_feature, a_t_prev, a_t_cands, h_t, c_t, ctx, ctx_mask)
        
        candidate_mask = utils.length2mask(cands_lengs, self.device)
        logits.masked_fill_(candidate_mask, -float('inf'))

        cpu_a_t = logits.max(1)[1].detach().cpu().numpy()
        for i, next_id in enumerate(cpu_a_t):
            if next_id == (len(obs[i]['candidates'])) \
                or next_id == self.ignore_id or ended[i]:
                cpu_a_t[i] = -1
        a_t_prev = a_t_cands[np.arange(batch_size), np.maximum(cpu_a_t, 0), :].detach()

        return logits, h_t, c_t, a_t_prev, img_feature, a_t_cands

    def save_model(self, model_save_path, **kwargs):
        output = kwargs
        output["encoder_state_dict"] = self.encoder.state_dict()
        output["decoder_state_dict"] = self.decoder.state_dict()

        torch.save(output, model_save_path)

    def load_model(self, model_load_path, ret:bool=True, cuda:int=0)->dict:
        checkpoint = torch.load(model_load_path, map_location=f"cuda:{cuda}")

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if ret: return checkpoint
    
    def reset_loss(self,):
        self.losses = []

    def trainable_params(self):
        param_list = []
        for _module in [self.encoder, self.decoder]:
            param_list += list(
                filter(lambda p: p.requires_grad, _module.parameters()))
        return param_list

    def train(self,):
        self.encoder.train()
        self.decoder.train()
    
    def eval(self,):
        self.encoder.eval()
        self.decoder.eval()
    
