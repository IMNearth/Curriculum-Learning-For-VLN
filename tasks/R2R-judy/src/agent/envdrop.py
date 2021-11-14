import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import src.utils as utils
import src.environ as environ
import src.model as M
from .base import BasicR2RAgent


class EnvDropAgent(BasicR2RAgent):
    """ Tan, H., Yu, L., & Bansal, M. (NAACL2019). 
        Learning to Navigate Unseen Environments: Back Translation with Environmental Dropout. 
        ArXiv, abs/1904.04195. """
    
    def __init__(self, 
        model_cfg,
        max_enc_len:int, 
        results_dir:str, 
        device:int,
        env:environ.R2RBatch, 
        tokenizer:utils.Tokenizer, 
        episode_len:int=20,
    ):
        super(EnvDropAgent, self).__init__(
            results_dir, device, env, tokenizer, episode_len=episode_len)

        self.cfg = model_cfg
        self.action_emb_size = self.cfg.ACT_EMB_SIZE
        self.max_enc_len = max_enc_len

        # Init the encoder + decoder
        self.encoder = M.EncoderLSTM(
            self.tokenizer.vocab_size(), 
            self.cfg.WORD_EMB_SIZE, 
            self.cfg.HIDDEN_SIZE, 
            padding_idx=utils.pad_idx, 
            drop_ratio=self.cfg.DROP_RATE, 
            bidirectional=self.cfg.ENC_BIDIRECTION, 
            num_layers=self.cfg.ENC_LAYERS, 
        )
        self.decoder = M.EnvDropDecoder(
            hidden_size=self.cfg.HIDDEN_SIZE, 
            drop_ratio=self.cfg.DROP_RATE,
            feat_drop_ratio=self.cfg.FEAT_DROP_RATE,
            action_embed_size=self.action_emb_size, 
            angle_feat_size=self.angle_feat_size,
            feature_size=self.feature_size,
        )
        self.critic = M.Critic(
            hidden_size=self.cfg.HIDDEN_SIZE, 
            drop_ratio=self.cfg.DROP_RATE, 
        )

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_id, reduction="none")
        
        # Logs
        self.logs = defaultdict(list)

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), self.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.ImageFeatures.make_angle_feat(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).to(self.device)

        img_feat = self._feature_variable(obs)      # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, img_feat, candidate_feat, candidate_leng

    def rollout(self, 
        train_ml=True, train_rl=False, train_cl=False, 
        reset=True, restart=False,
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
        if feedback != "sample": train_rl=False
    
        obs = self.env.reset(restart=restart) if reset else self.env.observe()
        batch_size = len(obs)

        if speaker is not None: # Trigger the self_train mode!
            noise = self.decoder.drop_env(torch.ones(self.img_feat_size, device=self.device))
            batch = self.env.batch.copy()
            speaker.env = self.env
            insts = speaker.infer_batch(featdropmask=noise)     # Use the same drop mask in speaker
            # Create fake environments with the generated instruction
            boss = np.ones((batch_size, 1), np.int64) * self.tokenizer.word_to_index['<BOS>']  # First word is <BOS>
            insts = np.concatenate((boss, insts), 1)
            for i, (datum, inst) in enumerate(zip(batch, insts)):
                if inst[-1] != self.tokenizer.word_to_index['<PAD>']: # The inst is not ended!
                    inst[-1] = self.tokenizer.word_to_index['<EOS>']
                datum.pop('instructions')
                datum.pop('instr_encoding')
                datum['instructions'] = self.tokenizer.decode_sentence(inst)
                datum['instr_encoding'] = inst
            obs = np.array(self.env.reset(batch=batch))
            pass
        
        # 1. Get the instruction representation
        seq_tensor, seq_mask, seq_length = self._instr_variable(obs)
        #    and encode the sequence
        ctx, h_t, c_t = self.encoder(seq_tensor, seq_length)

        # 2. Initialization
        # record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpointId'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        # for tracking state
        visited = [set() for _ in range(batch_size)]
        ended = np.array([False] * batch_size)
        # the reward shaping
        last_dist = np.array([ob['distance'] for ob in obs], np.float32)

        # 3. Logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        self.ml_loss = 0. if not train_cl else torch.zeros(batch_size, device=self.device)
        self.rl_loss = 0. if not train_cl else torch.zeros(batch_size, device=self.device)

        # 4. Start rolling out
        h_tilde = h_t
        for t in range(self.episode_len):
            input_a_t, img_feat, candidate_feat, candidate_leng = self.get_input_feat(obs)
            
            # Apply the env drop mask to the feat
            if speaker is not None:
                candidate_feat[..., :-self.angle_feat_size] *= noise
                img_feat[..., :-self.angle_feat_size] *= noise
            
            logits, (h_t, c_t), h_tilde = self.decoder(
                input_a_t, img_feat, candidate_feat, h_tilde, h_t, c_t, ctx, seq_mask, 
                already_dropfeat=(speaker is not None))
            
            hidden_states.append(h_t)
            
            # Mask outputs where agent can't move forward
            candidate_mask = utils.length2mask(candidate_leng, self.device)
            if avoid_cyclic:
                for i, ob in enumerate(obs):
                    visited[i].add(ob['viewpointId'])
                    for j, c in ob['candidates']:
                        if c['nextViewpointId'] in visited[i]:
                            candidate_mask[i][j] = 1
            logits.masked_fill_(candidate_mask, -float('inf'))

            # Supervised training
            # Get the ground truth action and Compute the loss
            target = self._teacher_action(obs, ended)
            if train_cl: self.ml_loss += self.criterion(logits, target) # vector
            else: self.ml_loss += self.criterion(logits, target).sum()  # value

            # Determine the action!
            if feedback == "teacher":  # teacher-forcing
                a_t = target
            elif feedback == "argmax": # student-forcing
                _, a_t = logits.max(1)
                log_probs = F.log_softmax(logits, 1)                            # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))  # Gather the log_prob for each batch
            elif feedback == "sample": # sample an action from model
                probs = F.softmax(logits, 1)
                c = torch.distributions.Categorical(probs)
                a_t = c.sample()
                policy_log_probs.append(c.log_prob(a_t))
                self.logs['entropy'].append(c.entropy().sum().item())           # For log
                entropys.append(c.entropy())                                    # For optimization
            else: raise NotImplementedError

            # Make action and get the new state
            cpu_a_t = a_t.detach().cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                # The last action is <stop> / Now at target point / already ended
                # Change the <stop> and ignore action to -1, which means the agent will not move
                if next_id == (len(obs[i]['candidates'])) or next_id == self.ignore_id or ended[i]:
                    cpu_a_t[i] = -1
            obs = self.move_and_observe(cpu_a_t, obs, traj)   

            # Calculate the mask and reward
            dist = np.array([ob['distance'] for ob in obs], np.float32)
            is_stop = (cpu_a_t == -1)
            reward = (
                is_stop * (2*(dist < 3)-1) * 2 + \
                (1 - is_stop) * np.sign(last_dist - dist)
            ) * (~ended)

            rewards.append(reward)
            masks.append(~ended)
            last_dist[:] = dist

            # Update the finished actions
            ended[:] = np.logical_or(ended, is_stop)
            if ended.all(): break
        
        # 5. Reinforcement learning: A2C
        if train_rl:
            # Last action in A2C
            input_a_t, img_feat, candidate_feat, candidate_leng = self.get_input_feat(obs)
            if speaker is not None:
                candidate_feat[..., :-self.angle_feat_size] *= noise
                img_feat[..., :-self.angle_feat_size] *= noise
            __, (last_h, __), __ = self.decoder(
                input_a_t, img_feat, candidate_feat, h_tilde, h_t, c_t, ctx, seq_mask, 
                already_dropfeat=(speaker is not None))
            
            # NOW, A2C!!!
            # Calculate the final discounted reward
            with torch.no_grad():
                # The value esti of the last state, remove the grad for safety
                last_value = self.critic(last_h).detach().cpu().numpy()
            discount_reward = (~ended) * last_value

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * self.cfg.GAMMA + rewards[t]     # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).to(self.device)
                r_ = Variable(torch.from_numpy(discount_reward), requires_grad=False).to(self.device)
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()
                
                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                cur_rl_loss = torch.zeros(batch_size, device=self.device)
                cur_rl_loss += (-policy_log_probs[t] * a_ * mask_)
                cur_rl_loss += (((r_ - v_) ** 2) * mask_) * 0.5                     # 1/2 L2 loss
                if feedback == "sample": cur_rl_loss += (- 0.01 * entropys[t] * mask_)
                if train_cl: self.rl_loss += cur_rl_loss    # vector
                else: self.rl_loss += cur_rl_loss.sum()     # value

                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())
                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if self.cfg.RL_NORMALIZE == 'total': self.rl_loss /= total              # this is the default choice
            elif self.cfg.RL_NORMALIZE == 'batch': self.rl_loss /= batch_size
            else: assert self.cfg.RL_NORMALIZE == 'none' 

        # 6. Save for optimization
        self.loss = {
            "ml_loss": self.ml_loss * self.cfg.ML_WEIGHT / batch_size if train_ml else .0,
            "rl_loss": self.rl_loss if train_rl else .0
        }

        # 7. Record the loss
        if train_cl and not restart:
            val = self.loss["ml_loss"] + self.loss["rl_loss"]
            if type(val) is float and val == 0.0: self.losses.append(0.0)       # For safety, it will be activated if no losses are added
            else: self.losses.append(val.sum().item())                          # otherwise, append its value
        
        return traj

    def running_state(self, h_t, c_t, h_tilde, **kwargs):
        return (h_t, c_t, h_tilde)

    def decode_obervation(self, obs, h_t, c_t, h_tilde, ctx, ctx_mask, ended, **kwargs):
        """ Agent dependent decoding process. """
        batch_size = len(obs)

        input_a_t, img_feat, candidate_feat, candidate_leng = self.get_input_feat(obs)

        logits, (h_t, c_t), h_tilde = self.decoder(
            input_a_t, img_feat, candidate_feat, h_tilde, h_t, c_t, ctx, ctx_mask, 
            already_dropfeat=False)
        
        candidate_mask = utils.length2mask(candidate_leng, self.device)
        logits.masked_fill_(candidate_mask, -float('inf'))

        return logits, h_t, c_t, h_tilde, img_feat, candidate_feat

    def save_model(self, model_save_path, **kwargs):
        output = kwargs
        output["encoder_state_dict"] = self.encoder.state_dict()
        output["decoder_state_dict"] = self.decoder.state_dict()
        output["critic_state_dict"] = self.critic.state_dict()

        torch.save(output, model_save_path)

    def load_model(self, model_load_path, ret:bool=True, cuda:int=0)->dict:
        checkpoint = torch.load(model_load_path, map_location=f"cuda:{cuda}")

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        if ret: return checkpoint
    
    def reset_loss(self,):
        self.losses = []
        self.logs = defaultdict(list)

    def trainable_params(self):
        param_list = []
        for _module in [self.encoder, self.decoder, self.critic]:
            param_list += list(
                filter(lambda p: p.requires_grad, _module.parameters()))
        return param_list

    def train(self,):
        self.encoder.train()
        self.decoder.train()
        self.critic.train()
    
    def eval(self,):
        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()
    



