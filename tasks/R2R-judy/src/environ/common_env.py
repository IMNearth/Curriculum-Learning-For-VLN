''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
import MatterSim
import math
import numpy as np
import json
import os
import random
import networkx as nx
import logging
logger = logging.getLogger("main.common_env")

import src.utils.misc as utils


ENV_ACTIONS = {
    'left':    (0,-1, 0), # left
    'right':   (0, 1, 0), # right
    'up':      (0, 0, 1), # up
    'down':    (0, 0,-1), # down
    'forward': (1, 0, 0), # forward
    '<end>':   (0, 0, 0), # <end>
    '<start>': (0, 0, 0), # <start>
    '<ignore>':(0, 0, 0)  # <ignore>
}

# ---------------------------
# --   Environment API     --
# ---------------------------

class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''
    
    image_w = 640
    image_h = 480
    vfov = 60

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print('\tThe feature size is %d' % self.feature_size)
                logger.info('\tThe feature size is %d' % self.feature_size)
        else:
            sys.exit('Image features not provided')
            logger.info('sys.exit: Image features not provided')
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        for __ in range(batch_size):
            sim = utils.M3DSimulator.new()
            self.sims.append(sim)
    
    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId  
    
    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)
    
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view 
            It will convert the action panoramic view action 'a_t'
            to equivalent egocentric view actions for the simulator.
        """
        for idx, at in enumerate(actions.tolist()):
            if at == -1: continue   # -1 is the <stop> action
            cand = obs[idx]['candidates'][at]
            sim = self.sims[idx]
            utils.M3DSimulator._move_to_location(
                sim=sim, 
                nextViewpointId=cand['nextViewpointId'], 
                absViewIndex=cand['absViewIndex'],
            )
            state = sim.getState()
            assert state.location.viewpointId == cand['nextViewpointId']
            if traj is not None:
                traj[idx]['path'].append(
                    (state.location.viewpointId, state.heading, state.elevation))


# ---------------------------
# --    Dataset    API     --
# ---------------------------

class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, splits=['train'], tokenizer=None, data_name:str="R2R", data_dir='tasks/R2R-judy/data'):
        print("\t... Initializing the R2RBatch ...")
        logger.info("\t... Initializing the R2RBatch ...")

        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        self.feature_size = self.env.feature_size
        self.tok = tokenizer

        self.data = []
        scans = []
        for split in splits:
            for item in utils.load_datasets([split], dataset=data_name, data_dir=data_dir):
                # Split multiple instructions into separate entries
                for j, instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    new_item['instr_encoding'], new_item['instr_length']\
                        = self.tok.encode_sentence(instr)
                    self.data.append(new_item)
                    scans.append(item['scan'])
        
        self.name = splits[0] if len(splits) > 0 else "FAKE"

        self.scans = set(scans)
        self.splits = splits
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils._static_loc_embeddings
        self.sim = utils.M3DSimulator.new()
        self.buffered_state_dict = {}

        print('\t... R2RBatch loaded with %d instructions, using splits: %s ...' % (len(self.data), ",".join(splits)))
        logger.info('\t... R2RBatch loaded with %d instructions, using splits: %s ...' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('\tLoading navigation graphs for %d scans' % len(self.scans))
        logger.info('\tLoading navigation graphs for %d scans' % len(self.scans))
        self.graphs = utils.load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, sort=True, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if tile_one:
            batch = [self.data[self.ix]] * self.batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+self.batch_size]
            if len(batch) < self.batch_size:
                random.shuffle(self.data)
                self.ix = self.batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += self.batch_size
        
        if sort and 'instr_length' in batch[0]: # by instructions length, high -> low:
            batch = sorted(batch, key=lambda item: item['instr_length'], reverse=True)

        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId)->int:
        ''' Determine next action on the shortest path to goal, 
            for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, imgFeature, scanId, viewpointId, viewId)->list:
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        
        long_id = "%s_%s" % (scanId, viewpointId)
        base_heading = (viewId % 12) * utils.angle_inc
        adj_dict = {}
        
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, -utils.angle_inc)
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)    # Turn right and look up
                else:
                    self.sim.makeAction(0, 1.0, 0)      # Turn right

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                img_feat = imgFeature[ix]
            
                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.ImageFeatures.make_angle_feat(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'scanId':scanId,
                            'absViewIndex': ix,
                            'nextViewpointId': loc.viewpointId,
                            'loc_heading': loc_heading,
                            'loc_elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((img_feat, angle_feat), -1)
                        }
            # store candidates
            candidates = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key] for key in
                    ['scanId', 'absViewIndex','nextViewpointId', 
                     'normalized_heading', 'loc_elevation', 'distance', 'idx']}
                for c in candidates]
        else: # use the buffer
            basic_candidates = self.buffered_state_dict[long_id]
            candidates = []
            for bc in basic_candidates:
                c = bc.copy()
                
                img_feat = imgFeature[c['absViewIndex']]
                c['loc_heading'] = c['normalized_heading'] - base_heading
                angle_feat = utils.ImageFeatures.make_angle_feat(
                    c['loc_heading'], c['loc_elevation'])
                c['feature'] = np.concatenate((img_feat, angle_feat), -1)
                c.pop('normalized_heading')
                # keys: scanId, absViewIndex, nextViewpointId, 
                #       loc_heading, loc_elevation, distance, idx, viewFeat
                candidates.append(c)
        
        return candidates

    def observe(self)->list:
        obs = []
        for i, (img_feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]

            # Full features
            candidates = self.make_candidate(
                img_feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature_with_angel = np.concatenate((img_feature, self.angle_feature[state.viewIndex]), -1)
            obs.append({
                'instr_id' : item['instr_id'],                      # instruction id
                'scan' : state.scanId,                              # scan id
                'viewpointId' : state.location.viewpointId,         # current viewpoint id, each viewpoint has 36 views
                'viewIndex' : state.viewIndex,                      # current view index
                'heading' : state.heading,                          # current view heading
                'elevation' : state.elevation,                      # current view elevation
                'feature' : feature_with_angel,                     # current view feature (img + angel)
                'candidates': candidates,                           # navigable viewpoints from all 36 views
                'navigableLocations' : state.navigableLocations,    # navigable locations from current view
                'instructions' : item['instructions'],              # a single instruction
                'teacher' : self._shortest_path_action(state, item['path'][-1]),    # nearest viewpoint id
                'path_id' : item['path_id']                         # path id
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']  # instruction encoded token ids
            if 'instr_length' in item:
                obs[-1]['instr_length'] = item['instr_length']      # instruction length
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, restart=False, **kwargs)->list:
        ''' Load a new minibatch / episodes. '''
        if not restart:     # Load the next mini-batch
            if batch is None:       # Allow the user to explicitly define the batch
                self._next_minibatch(**kwargs)
            else:
                if inject:          # Inject the batch into the next minibatch
                    self._next_minibatch(**kwargs)
                    self.batch[:len(batch)] = batch
                else:               # Else set the batch to the current batch
                    self.batch = batch
        else: pass          # Do nothing, just re-use the current batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self.observe()

    def step(self, actions:np.ndarray, obs:list, traj:list)->list:
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions, obs, traj)
        return self.observe()

    def get_statistics(self)->dict:
        ''' Compute average instr length and average path elngth '''
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats


# ---------------------------
# --    Dataset    API     --
# ---------------------------

class RxRBatch(R2RBatch):

    def __init__(self, feature_store, batch_size=100, splits=['train'], tokenizer=None, data_name:str="RxR", data_dir='tasks/R2R-judy/data/RxR-en'):
        print("\t... Initializing the RxRBatch ...")
        logger.info("\t... Initializing the RxRBatch ...")

        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        self.feature_size = self.env.feature_size
        self.tok = tokenizer

        self.data = []
        scans = []
        for split in splits:
            for item in utils.load_datasets([split], dataset=data_name, data_dir=data_dir):
                new_item = dict(item)
                del new_item['instruction']
                del new_item['instruction_id']
                new_item['instructions'] = item['instruction']
                new_item['instr_id'] = item['instruction_id']
                new_item['instr_encoding'], new_item['instr_length']\
                        = self.tok.encode_sentence(item['instruction'])
                self.data.append(new_item)
                scans.append(item['scan'])
        
        self.name = splits[0] if len(splits) > 0 else "FAKE"

        self.scans = set(scans)
        self.splits = splits
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils._static_loc_embeddings
        self.sim = utils.M3DSimulator.new()
        self.buffered_state_dict = {}

        print('\t... RxRBatch loaded with %d instructions, using splits: %s ...' % (len(self.data), ",".join(splits)))
        logger.info('\t... RxRBatch loaded with %d instructions, using splits: %s ...' % (len(self.data), ",".join(splits)))


