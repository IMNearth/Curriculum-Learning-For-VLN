''' Evaluation of Agent trajectories '''

import json
from collections import defaultdict
import networkx as nx
import numpy as np

import src.utils as utils

class Evaluation(object):
    ''' 
    Results submission format:  
        [
            {
                'instr_id': string, 
                'trajectory':[(viewpoint_id, heading_rads, elevation_rads),]
            },
        ] 
    '''

    def __init__(self, splits, data_name="R2R", data_dir="tasks/R2R-judy/data"):
        self.error_margin = 3.0
        self.splits = splits
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        self.dataset = data_name

        if data_name == "RxR": self.instr2path = {}

        for item in utils.load_datasets(splits, dataset=data_name, data_dir=data_dir):
            self.gt[item['path_id']] = item
            self.scans.append(item['scan'])
            if data_name in ["R2R", "CLR2R", "R4R"]: 
                self.instr_ids += ['%d_%d' % (item['path_id'],i) for i in range(3)]
            elif data_name == "RxR": 
                self.instr_ids.append(item['instruction_id'])
                self.instr2path[item['instruction_id']] = item['path_id']
            else: raise NotImplementedError
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)

        self.graphs = utils.load_nav_graphs(self.scans)
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
    
    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule).
            The path contains [view_id, angle, vofv] '''
        if self.dataset == "RxR": gt = self.gt[self.instr2path[instr_id]]
        else: gt = self.gt[int(instr_id.split('_')[0])]
        
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt['scan'], goal, path)

        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores['trajectory_steps'].append(len(path)-1)
        # more metrics
        dtw_worker = utils.DTW(distance=self.distances[gt['scan']], threshold=self.error_margin)
        predicted_path = list(zip(*path))[0]
        ndtw, sdtw = dtw_worker(predicted_path, gt['path'], metric=['ndtw', 'sdtw'])
        self.scores['ndtws'].append(ndtw)
        self.scores['sdtws'].append(sdtw)
        # more more metrics
        cls_worker = utils.CLS(distance=self.distances[gt['scan']], threshold=self.error_margin)
        cls_score = cls_worker(predicted_path, gt['path'])
        self.scores['clss'].append(cls_score) 

        # Work out the length of the path in meters
        distance = 0
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        is_success = self.distances[gt['scan']][final_position][goal] < self.error_margin

        if self.splits == ['test']:
            self.scores['success_path_length'].append(0)
        else:
            self.scores['success_path_length'].append(
                is_success * self.distances[gt['scan']][start][goal] / \
                    max(self.distances[gt['scan']][start][goal], distance))

    def score(self, output):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        # unique instrunction ids in ground truth
        instr_ids = set(self.instr_ids)

        # 1. read in saved trajectory
        if isinstance(output, str):
            output_file = output
            with open(output_file) as f:
                for item in json.load(f):
                    # Check against expected ids
                    if item['instr_id'] in instr_ids:
                        instr_ids.remove(item['instr_id'])
                        self._score_item(item['instr_id'], item['trajectory'])
        elif isinstance(output, list):
            output_results = output
            for item in output:
                # Check against expected ids
                if item['instr_id'] in instr_ids:
                    instr_ids.remove(item['instr_id'])
                    self._score_item(item['instr_id'], item['trajectory'])
        else: raise NotImplementedError
        assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                       % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
        assert len(self.scores['nav_errors']) == len(self.instr_ids)

        # 2. create summary dict
        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'lengths': np.average(self.scores['trajectory_lengths']),
            'spl': np.average(self.scores['success_path_length']),
            'ndtw': np.average(self.scores['ndtws']),
            'sdtw': np.average(self.scores['sdtws']),
            'cls': np.average(self.scores['clss']),
        }

        # 3. compute success rate and oracle success rate
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        score_summary['success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))
        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))
        
        return score_summary, self.scores
