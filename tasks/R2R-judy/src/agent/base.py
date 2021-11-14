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
import logging

logger = logging.getLogger("main.base_agent")


class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_dir):
        self.env = env
        self.results_save_dir = results_dir
        random.seed(1)
        self.results = {}       # record final trajectories, (key, value) = (instr_id, path)
        self.losses = []        # record the losses in training process
    
    def write_results(self, split="train"):
        ''' Write results into json file. '''
        result_path = os.path.join(self.results_save_dir, "%s.json" % split)
        outputs = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(result_path, 'w') as f:
            json.dump(outputs, f)
    
    def get_results(self,):
        ''' Extract the results into a list. 
            Each element should be a dict like:
                {
                    "instr_id": ... ,
                    "trajectory": ..., 
                    etc
                }
        '''
        # This is a naive version. 
        outputs = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        return outputs

    def rollout(self, **args):
        ''' Return a list of dicts 
            containing instr_id:'xx', 
                       path:[(viewpointId, heading_rad, elevation_rad)]  
        '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]
    
    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for __ in range(iters):
                for traj in self.rollout(**kwargs):
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break

###############################
#       Basic Version         #
###############################


class BasicR2RAgent(BaseAgent):
    ''' An unified R2R basic agent. '''

    ignore_id = -1

    def __init__(self, 
        results_dir:str, 
        device:torch.device,
        env:environ.R2RBatch, 
        tokenizer:utils.Tokenizer, 
        img_feat_size:int=2048,
        angle_feat_size:int=128, 
        episode_len:int=20
    ):
        super(BasicR2RAgent, self).__init__(env, results_dir)

        self.device = device
        self.tokenizer = tokenizer
        self.episode_len = episode_len

        # record size
        self.img_feat_size = img_feat_size
        self.angle_feat_size = angle_feat_size
        self.feature_size = self.img_feat_size + self.angle_feat_size # 2048 + 128

    def _instr_variable(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). 
            
            Returns: torch.Variable(batch_size, max_seq_len)
                     torch.Bool(batch_size, seq_lengths[0])
                     list, list
        '''
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.array([ob['instr_length'] for ob in obs])
        # seq_lengths = np.argmax(seq_tensor == utils.pad_idx, axis=1)
        # seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]          # Full length
        # assert seq_lengths == np.array([ob['instr_length'] for ob in obs])

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Batch data are already sorted
        # # Sort sequences by lengths
        # seq_lengths, perm_idx = seq_lengths.sort(0, True)           # True -> descending
        # sorted_tensor = seq_tensor[perm_idx]
        seq_tensor = seq_tensor[:, :seq_lengths[0]]                   # seq_lengths[0] is the Maximum length
        mask = (seq_tensor == utils.pad_idx)                          # True if we need to mask the item

        return Variable(seq_tensor, requires_grad=False).long().to(self.device), \
               mask.bool().to(self.device), seq_lengths

    def _feature_variable(self, obs, num_views:int=36):
        ''' Extract precomputed features into variable. '''
        _shape = (len(obs), num_views, self.feature_size)
        features = np.empty(_shape, dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat with angle info
        return Variable(torch.from_numpy(features), requires_grad=False).to(self.device)

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidates']) + 1 for ob in obs]      # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidates']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidates']):
                candidate_feat[i, j, :] = c['feature']                  # Image feat with angle info
        return torch.from_numpy(candidate_feat).to(self.device), candidate_leng
    
    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                              # Just ignore this index
                a[i] = self.ignore_id
            else:
                for k, candidate in enumerate(ob['candidates']):
                    if candidate['nextViewpointId'] == ob['teacher']: # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpointId']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidates'])
        return torch.from_numpy(a).to(self.device)
    
    def move_and_observe(self, a_t, last_obs, traj=None):
        return self.env.step(a_t, last_obs, traj)

    def _dijkstra(self, max_candidates):
        """
        The dijkstra algorithm.
        Was called beam search to be consistent with existing work.
        But it actually finds the Exact K paths with smallest listener log_prob.
        :return:
        [{
            "scan": XXX
            "instr_id":XXX,
            'instr_encoding": XXX
            'dijk_path': [v1, v2, ..., vn]      (The path used for find all the candidates)
            "paths": {
                    "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        }]
        ------------
        github@airsplay/R2R-EnvDrop/r2r_src/agent.py#L452
        """
        def make_state_id(viewpoint, action):       # Make state id
            return "%s_%s" % (viewpoint, str(action))
        def decompose_state_id(state_id):           # Make state id
            viewpoint, action = state_id.split("_")
            action = int(action)
            return viewpoint, action
        
        obs = self.env.reset()                      # Get first obs
        
        batch_size = len(obs)
        results = [{                                # Prepare the state id
            "scan": ob['scan'],
            "instr_id": ob['instr_id'],
            "instr_encoding": ob["instr_encoding"],
            "dijk_path": [ob['viewpointId']],
            "paths": []
        } for ob in obs]

        # 1. Get the instruction representation
        #    and encode the sequence:
        seq_tensor, seq_mask, seq_length = self._instr_variable(obs)
        ctx, h_t, c_t = self.encoder(seq_tensor, seq_length)
        #    Note that ctx_mask = seq_mask.
        #    Also note that encoding process is same among all agents.

        a_t_prev = Variable(torch.zeros(\
            batch_size, self.action_emb_size, device=self.device),\
            requires_grad=False)

        # 2. Init dijkstra graph states:
        id2state = [{
            make_state_id(ob['viewpointId'], -95):{
                "next_viewpoint": ob['viewpointId'],
                "running_state": self.running_state(
                    h_t=h_t[i], c_t=c_t[i], h_tilde=h_t[i], a_t_prev=a_t_prev[i]),
                "location": (ob['viewpointId'], ob['heading'], ob['elevation']),
                "from_state_id": None,
                "feature": None,
                "score": 0,
                "scores": [],
                "actions": [],
            }
        }  for i, ob in enumerate(obs)]             # -95 is the start point
        
        # 3. Prepare recorders
        visited = [set() for _ in range(batch_size)]
        finished = [set() for _ in range(batch_size)]
        graphs = [utils.FloydGraph() for _ in range(batch_size)]    # For the navigation path
        ended = np.array([False] * batch_size)      # whether to stop search of sample i

        # 4. Dijkstra algorithm
        for _ in range(500):
            # (1) Get the state with smallest score for each batch:
            # If the batch is not ended, find the smallest item;
            # else use a random item from the dict (always exists).
            smallest_idXstate = [
                max(
                    ((state_id, state) for state_id, state in id2state[i].items() if state_id not in visited[i]),
                    key=lambda item: item[1]['score']
                ) if not ended[i] else next(iter(id2state[i].items()))
                for i in range(batch_size)
            ]

            # (2) Set the visited and the end seqs
            tmp_ended = []
            for i, (state_id, state) in enumerate(smallest_idXstate):
                assert (ended[i]) or (state_id not in visited[i])
                if not ended[i]:
                    viewpoint, action = decompose_state_id(state_id)
                    visited[i].add(state_id)
                    if action == -1:    # where ignore_id = -1
                        tmp_ended.append(True)
                        finished[i].add(state_id)
                        if len(finished[i]) >= max_candidates:      # Get enough candidates
                            ended[i] = True
                    else: tmp_ended.append(False)
                else: tmp_ended.append(True)

            # (3) Gather the running state in the batch
            h_ts, c_ts, apORhs = zip(*(idXstate[1]['running_state'] for idXstate in smallest_idXstate))
            h_t, c_t, apORh = torch.stack(h_ts), torch.stack(c_ts), torch.stack(apORhs), 

            # (4) Recover the env and gather the feature
            for i, (state_id, state) in enumerate(smallest_idXstate):
                next_viewpoint = state['next_viewpoint']
                scan = results[i]['scan']
                from_viewpoint, heading, elevation = state['location']
                # Heading, elevation is not used in panoramic
                self.env.env.sims[i].newEpisode(scan, next_viewpoint, heading, elevation)
            obs = self.env.observe()

            # (5) Update the floyd graph:
            # Only used to shorten the navigation length
            # Will not effect the result
            for i, ob in enumerate(obs):
                viewpoint = ob['viewpointId']
                if not graphs[i].visited(viewpoint):    # Update the Graph
                    for c in ob['candidates']:
                        next_viewpoint = c['nextViewpointId']
                        dis = self.env.distances[ob['scan']][viewpoint][next_viewpoint]
                        graphs[i].add_edge(viewpoint, next_viewpoint, dis)
                    graphs[i].update(viewpoint)
                results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], viewpoint))   

            # (6) Run one decoding step
            #  which depneds on navigation models.
            masked_logits, h_t, c_t, apORh, f_t, candidate_feat = self.decode_obervation(
                obs, h_t, c_t, apORh, ctx, seq_mask, ended=tmp_ended
            )
            log_probs = F.log_softmax(masked_logits, 1)             # Calculate the log_prob here

            for i, ob in enumerate(obs):
                current_viewpoint, candidate = ob['viewpointId'], ob['candidates']
                current_state_id, current_state = smallest_idXstate[i]
                old_viewpoint, from_action = decompose_state_id(current_state_id)
                assert ob['viewpointId'] == current_state['next_viewpoint']
                
                # If the action is <end> or the batch is ended, skip it
                if from_action == -1 or ended[i]: continue
                
                for j in range(len(ob['candidates']) + 1):                   # +1 to include the <end> action
                    # score + log_prob[action]
                    modified_log_prob = log_probs[i][j].detach().cpu().item() 
                    new_score = current_state['score'] + modified_log_prob
                    if j < len(candidate):                                  # A normal action
                        next_id = make_state_id(current_viewpoint, j)
                        next_viewpoint = candidate[j]['nextViewpointId']
                        trg_point = candidate[j]['absViewIndex']
                        heading = (trg_point % 12) * math.pi / 6
                        elevation = (trg_point // 12 - 1) * math.pi / 6
                        location = (next_viewpoint, heading, elevation)
                    else:                                                   # The end action
                        next_id = make_state_id(current_viewpoint, -1)      # action is -1
                        next_viewpoint = current_viewpoint                  # next viewpoint is still here
                        location = (current_viewpoint, ob['heading'], ob['elevation'])
                    # Update
                    if next_id not in id2state[i] or new_score > id2state[i][next_id]['score']:
                        id2state[i][next_id] = {
                            "next_viewpoint": next_viewpoint,
                            "location": location,
                            "running_state": self.running_state(h_t[i], c_t[i], apORh[i]),
                            "from_state_id": current_state_id,
                            "feature": (f_t[i].detach().cpu(), candidate_feat[i][j].detach().cpu()),
                            "score": new_score,
                            "scores": current_state['scores'] + [modified_log_prob],
                            "actions": current_state['actions'] + [len(candidate)+1],
                        }
                
                # The active state is zero after the updating, then setting the ended to True
                if len(visited[i]) == len(id2state[i]):             # It's the last active state
                    ended[i] = True
            
            # (7) End?
            if ended.all(): break
            
        # 5. Move back to the start point
        for i in range(batch_size):
            results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], results[i]['dijk_path'][0]))
        
        """
            "paths": [
                {
                    "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                }
            ...]
        """
        # 6. Gather the Path
        for i, result in enumerate(results):
            assert len(finished[i]) <= max_candidates
            for state_id in finished[i]:
                path_info = {
                    "trajectory": [],
                    "action": [],
                    "listener_scores": id2state[i][state_id]['scores'],
                    "listener_actions": id2state[i][state_id]['actions'],
                    "visual_feature": []
                }
                viewpoint, action = decompose_state_id(state_id)
                while action != -95:
                    state = id2state[i][state_id]
                    path_info['trajectory'].append(state['location'])
                    path_info['action'].append(action)
                    path_info['visual_feature'].append(state['feature'])
                    state_id = id2state[i][state_id]['from_state_id']
                    viewpoint, action = decompose_state_id(state_id)
                state = id2state[i][state_id]
                path_info['trajectory'].append(state['location'])
                for need_reverse_key in ["trajectory", "action", "visual_feature"]:
                    path_info[need_reverse_key] = path_info[need_reverse_key][::-1]
                result['paths'].append(path_info)

        return results

    def beam_rollout(self, speaker, beam_size):
        """
        :param speaker: The speaker to be used in searching.
        :return:
        {
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "speaker_scores": [log_prob_word1, log_prob_word2, ..., ],
            }]
        }
        ------------
        github@airsplay/R2R-EnvDrop/r2r_src/agent.py#L699
        """
        results = self._dijkstra(beam_size)         # get the sampled path using dijkstra

        # Compute the speaker scores:
        for result in results:
            # record path lengths
            lengths = []
            for path in result['paths']:
                assert len(path['trajectory']) == (len(path['visual_feature']) + 1)
                lengths.append(len(path['visual_feature']))
            if len(lengths) == 0: 
                logger.info("Zero result is Trigged! {}-{}".fromat(result["scan"], result["instr_id"]))
                continue
            max_len = max(lengths)
            # create feature tensors
            num_paths = len(result['paths'])
            img_feats = torch.zeros(num_paths, max_len, 36, self.feature_size)
            can_feats = torch.zeros(num_paths, max_len, self.feature_size)
            for j, path in enumerate(result['paths']):
                for k, feat in enumerate(path['visual_feature']):
                    img_feat, can_feat = feat
                    img_feats[j][k] = img_feat
                    can_feats[j][k] = can_feat
            img_feats, can_feats = img_feats.to(self.device), can_feats.to(self.device)
            features = ((img_feats, can_feats), lengths)
            # compute speaker scores
            insts = np.array([result['instr_encoding'] for _ in range(num_paths)])
            seq_lengths = np.argmax(insts == self.tokenizer.word_to_index['<EOS>'], axis=1)   # len(seq + 'BOS') == len(seq + 'EOS')
            insts = torch.from_numpy(insts).to(self.device)
            speaker_scores = speaker.teacher_forcing(train=True, features=features, insts=insts, for_listener=True)
            for j, path in enumerate(result['paths']):
                path.pop("visual_feature")
                path['speaker_scores'] = -speaker_scores[j].detach().cpu().numpy()[:seq_lengths[j]]
        return results

    def beam_search(self, speaker, beam_size:int=30):
        self.eval()

        looped = False
        self.results = {}
        while True:
            for traj in self.beam_rollout(speaker, beam_size):
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj
            if looped:
                break

    def running_state(self, **kwargs):
        """
        Should be overrided to enable beam search!
        """
        pass

    def decode_obervation(self, **kwargs):
        """
        Agent dependent decoding process.
        Should be overrided to enable beam search!
        """
        # TODO: implement this
        raise NotImplementedError


###############################
#       Sanity Check          #
###############################
class TestAgent(BasicR2RAgent):
    ''' Test if the warpped functions are right.'''

    def __init__(self, 
        results_dir:str, 
        device:int,
        env:environ.R2RBatch, 
        tokenizer:utils.Tokenizer, 
        img_feat_size:int=2048,
        angle_feat_size:int=128, 
        episode_len:int=20,
    ):
        super(TestAgent, self).__init__(
            results_dir, device, env, tokenizer, img_feat_size, angle_feat_size, episode_len)

        self.model = None
        
    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None, feedback="teacher"):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether to use RL in training
        :param reset:       Reset the environment
        :param speaker:     Speaker used in back translation.
                            If the speaker is not None, use back translation.
                            O.w., normal training
        """
        obs = self.env.reset() if reset else self.env.observe()
        batch_size = len(obs)

        # 1. Get the instruction representation
        seq_tensor, seq_mask, seq_length = self._instr_variable(obs)
        # print('seq_tensor', seq_tensor.shape, seq_tensor.dtype)
        # print('seq_mask', seq_mask.shape, seq_mask.dtype)
        # print(seq_mask)
        # print('seq_length', seq_length)

        # 2. Init the reward shaping
        # The init distance from the view point to the target
        last_dist = np.array([ob['distance'] for ob in obs], np.float32)
        # print('last_dist', last_dist)
        
        # 3. Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpointId'], ob['heading'], ob['elevation'])]
        } for ob in obs]

        # 4. Initialization the tracking state
        ended = np.array([False] * batch_size)
        rewards = []

        # 5. Start rolling out
        for t in range(self.episode_len):

            # teacher feedback
            target = self._teacher_action(obs, ended)

            if feedback == "teacher": a_t = target
            elif feedback == "argmax": pass
            elif feedback == "sample": pass
            else: raise NotImplementedError

            cpu_a_t = a_t.detach().cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                # The last action is <stop> / Now at target point / already ended
                if next_id == (len(obs[i]['candidates'])) or next_id == self.ignore_id or ended[i]:
                    cpu_a_t[i] = -1             # Change the <stop> and ignore action to -1, which means <end>
            
            # 6. Make action and get the new state
            obs = self.move_and_observe(cpu_a_t, obs, traj)
            
            # 7. Calculate the mask and reward
            dist = np.array([ob['distance'] for ob in obs], np.float32)
            is_stop = (cpu_a_t == -1)
            reward = (
                is_stop * np.sign(last_dist - dist) + \
                (1 - is_stop) * (2*(dist < 3)-1) * 2
            ) * (1-ended)

            rewards.append(reward)
            last_dist[:] = dist

            # 8. Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, is_stop)
            if ended.all(): break

        return traj

