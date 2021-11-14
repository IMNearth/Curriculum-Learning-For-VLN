''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
import MatterSim
import math
import numpy as np
import json
import os
import ast
import random
from collections import defaultdict
import networkx as nx
import logging
logger = logging.getLogger("main.sub_instr_env")


import src.utils.misc as utils
from .common_env import R2RBatch, EnvBatch


# ---------------------------
# --    Curriculum  API    --
# ---------------------------

class CLR2RBatch(R2RBatch):
    ''' Implements the Room to Room navigation task, 
        this case using FGR2R data and discretized viewpoints + pretrained features. '''
    
    def __init__(self, feature_store, batch_size=100, c_rate=0.8, tokenizer=None, data_dir='tasks/R2R-judy/data/CLR2Rv3'):
        print("\t... Initializing the CLR2RBatch ...")
        logger.info("\t... Initializing the CLR2RBatch ...")
    
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        self.feature_size = self.env.feature_size
        self.tok = tokenizer
        self.c_rate = c_rate
    
        self.splits = []
        self.data = []
        self.curriculum_data = defaultdict(list)
        scans = []

        for k in range(1, 6):
            split = f"train_round[{k}]_v3"
            for item in utils.load_datasets([split], dataset="CLR2R", data_dir=data_dir):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    new_item['instr_encoding'], new_item['instr_length']\
                        = self.tok.encode_sentence(instr)

                    self.data.append(new_item)
                    self.curriculum_data[f"round_{k}"].append(new_item)
                    scans.append(item['scan'])
            self.splits.append(split)
            print("\t\t {}: {} items.".format(split, len(self.curriculum_data[f"round_{k}"])))
            logger.info("\t\t {}: {} items.".format(split, len(self.curriculum_data[f"round_{k}"])))

        self.name = "train"
        self.scans = set(scans)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils._static_loc_embeddings
        self.sim = utils.M3DSimulator.new()
        self.buffered_state_dict = {}

        self._init_curriculum()

        print('\t... CLR2RBatch loaded with %d instructions, using splits: %s ...' % (len(self.data), ",".join(self.splits)))
        logger.info('\t... CLR2RBatch loaded with %d instructions, using splits: %s ...' % (len(self.data), ",".join(self.splits)))    

    def _init_curriculum(self,):
        """ Initialize necessary variables for CL. """
        self.a = np.zeros(len(self.data), dtype=np.float32)

        self.item2idx = {}
        for __, (key, data) in enumerate(self.curriculum_data.items()):
            for item in data:
                current_idx = len(self.item2idx)
                self.item2idx[item['instr_id']] = current_idx
                self.a[current_idx] = int(key[-1])
        
        self.c = self.a.sum() * self.c_rate
    
    def __len__(self):
        return len(self.data)

    def index(self, item):
        return self.item2idx[item['instr_id']]

    @property
    def cur_batch_index(self, ):
        return list(map(lambda x: self.item2idx[x['instr_id']], self.batch))
