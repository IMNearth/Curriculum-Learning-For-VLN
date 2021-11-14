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


class SelfMonitorAgent(BasicR2RAgent):
    """ Self-Monitoring Navigation Agent via Auxiliary Progress Estimation
        Ma, Chih-Yao et al. (ICLR), 2019 """
    
    def __init__(self, 
        model_cfg,
        max_enc_len:int, 
        results_dir:str, 
        device:int,
        env:environ.R2RBatch, 
        tokenizer:utils.Tokenizer, 
        episode_len:int=20,
    ):
        super(SelfMonitorAgent, self).__init__(
            results_dir, device, env, tokenizer, episode_len=episode_len)

        self.cfg = model_cfg
        self.action_emb_size = self.feature_size
        self.max_enc_len = max_enc_len

        self.encoder = M.EncoderLSTM(
            self.tokenizer.vocab_size(), 
            self.cfg.WORD_EMB_SIZE, 
            self.cfg.HIDDEN_SIZE, 
            padding_idx=utils.pad_idx, 
            drop_ratio=self.cfg.DROP_RATE, 
            bidirectional=self.cfg.ENC_BIDIRECTION, 
            num_layers=self.cfg.ENC_LAYERS, 
        )

        self.decoder = M.MonitorDecoder(
            rnn_hidden_size=self.cfg.HIDDEN_SIZE, 
            drop_ratio=self.cfg.DROP_RATE, 
            max_enc_len=self.max_enc_len, 
            mlp_dims=self.cfg.MLP_HIDDEN, 
            action_embed_size=self.action_emb_size,
            feature_size=self.feature_size
        )

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_id)
        self.curriculum_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_id, reduction="none")
        self.mseloss = nn.MSELoss()
        self.curriculum_mseloss = nn.MSELoss(reduction="none")
    
    def _instr_variable(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). 
            
            Returns: torch.Variable(batch_size, max_seq_len)
                     torch.Bool(batch_size, seq_lengths[0])
                     list, list
        '''
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.array([ob['instr_length'] for ob in obs])

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Batch data are already sorted
        # True if we need to mask the item
        mask = (seq_tensor == utils.pad_idx)

        return Variable(seq_tensor, requires_grad=False).long().to(self.device), \
               mask.bool().to(self.device), seq_lengths
    
    def rollout(self, 
        train_ml=True, train_cl=False, reset=True, restart=False, 
        lamb:float=0.5, speaker=None, avoid_cyclic=False, feedback="sample"
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
        
        # 4. Initialize other recoding variables
        ended = np.array([False] * batch_size)
        visited = [set() for _ in range(batch_size)]
        start_dist = np.array([ob['distance'] for ob in obs], np.float32)
        cur_dist = np.array([ob['distance'] for ob in obs], np.float32)
        # Machine learning loss, ml_loss = action_selection_loss + progress_loss
        self.ml_loss = 0.           # gradient backward
        self.progress_loss = 0.     # purely recording
        
        # 5. Start rolling out
        for t in range(self.episode_len):
            #(1) Get candidates and candidate_mask
            a_t_cands, cands_lengs = self._candidate_variable(obs)
            candidate_mask = utils.length2mask(cands_lengs, self.device)
            if avoid_cyclic:
                for i, ob in enumerate(obs):
                    visited[i].add(ob['viewpointId'])
                    for j, c in ob['candidates']:
                        if c['viewpointId'] in visited[i]:
                            candidate_mask[i][j] = 1
            
            #(2) Run the decoder
            (logits, cur_prog_val), (h_t, c_t), __ = self.decoder(
                None, a_t_prev, a_t_cands, h_t, c_t, ctx, seq_mask, candidate_mask)
            #(3) Mask outputs where agent can't move forward
            logits.masked_fill_(candidate_mask, -float('inf'))
            
            #(4) Get the ground truth action and Compute the loss
            target = self._teacher_action(obs, ended)
            if not train_cl: cur_action_loss = self.criterion(logits, target)
            else: cur_action_loss = self.curriculum_criterion(logits, target)
            if t == 0: cur_loss = cur_action_loss
            else: # action_loss + progress_monitor_loss
                prog_target = (start_dist - cur_dist) / start_dist
                prog_target[cur_dist<=3.0] = 1.0
                prog_target[ended] = cur_prog_val.detach().cpu().numpy()[ended]
                prog_target = torch.from_numpy(prog_target).to(self.device)
                if not train_cl: 
                    cur_progress_loss = self.mseloss(cur_prog_val, prog_target)
                    self.progress_loss += cur_progress_loss.item()
                else:
                    cur_progress_loss = self.curriculum_mseloss(cur_prog_val, prog_target)
                    self.progress_loss += cur_progress_loss.mean().item()
                cur_loss = lamb * cur_progress_loss + (1-lamb) * cur_action_loss
            self.ml_loss += cur_loss
            
            #(5) Determine the action!
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
            
            # (6) Make action and get the new state, also trajectories are recorded ...
            obs = self.move_and_observe(cpu_a_t, obs, traj)

            # (7) Update useful tensors / arrays
            is_stop = (cpu_a_t == -1)
            cur_dist[:] = np.array([ob['distance'] for ob in obs], np.float32)
            ended[:] = np.logical_or(ended, is_stop)
            a_t_prev = a_t_cands[np.arange(batch_size), np.maximum(cpu_a_t, 0), :].detach()

            if ended.all(): break
    
        # 6. Record the loss
        if not train_cl and not restart: self.losses.append(self.ml_loss.item())
        if not restart: self.progress_losses.append(self.progress_loss)
    
        return traj

    def running_state(self, h_t, c_t, a_t_prev, **kwargs):
        return (h_t, c_t, a_t_prev)

    def decode_obervation(self, obs, h_t, c_t, a_t_prev, ctx, ctx_mask, ended, **kwargs):
        """ Agent dependent decoding process. """
        batch_size = len(obs)

        img_feature = self._feature_variable(obs)  # (batch, num_views, feature_size)
        a_t_cands, cands_lengs = self._candidate_variable(obs)
        candidate_mask = utils.length2mask(cands_lengs, self.device)

        (logits, __), (h_t, c_t), __ = self.decoder(
            None, a_t_prev, a_t_cands, h_t, c_t, ctx, ctx_mask, candidate_mask)
        
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
        self.progress_losses = []

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
