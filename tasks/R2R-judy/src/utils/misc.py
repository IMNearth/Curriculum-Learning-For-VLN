''' Utils for io, language, connectivity graphs etc '''

import os
import sys
import re
sys.path.append('build')
import MatterSim
import string
import json
import time
import math
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
import prettytable as pt
import logging
logger = logging.getLogger("main.utils")


# padding, unknown word, end of sentence, begin of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>', '<BOS>']
pad_idx = base_vocab.index('<PAD>')
unk_idx = base_vocab.index('<UNK>')
eos_idx = base_vocab.index('<EOS>')
bos_idx = base_vocab.index('<BOS>')


angle_inc = math.pi / 6.


#######################
##     DATA LOAD     ##
#######################

def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3], 
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def load_datasets(splits, dataset:str="R2R", data_dir="tasks/R2R-judy/data"):
    """ should be changed based on your task and data """
    data = []
    for split in splits:
        with open('%s/%s_%s.json' % (data_dir, dataset, split)) as f:
            data += json.load(f)
    return data


def load_room_connectivity(scans):
    ''' Load ROOM connectivity graph for each scan '''
    
    connectivity = {}
    for scan in scans:
        with open('room_connectivity/%s_panorama_to_region.txt' % scan) as f:
            lines = list(map(lambda x: x.strip().split(), f.readlines()))
        
        room_info = defaultdict(list)
        for __, viewpointId, room_idx, room_type in lines:
            room_name = "_".join([room_idx, room_type])
            room_info[room_name].append(viewpointId)
        
        connectivity[scan] = room_info
    
    return connectivity


#######################
##     TOKENIZER     ##
#######################

class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character

    def __init__(self, vocab:list=None, encoding_length:int=20):
        self.vocab = vocab
        self.encoding_length = encoding_length
        self.word_to_index = {}
        self.index_to_word = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i
            # set default value
            new_w2i = defaultdict(lambda: self.word_to_index['<UNK>'])
            new_w2i.update(self.word_to_index)
            self.word_to_index = new_w2i
            # store reverse dict
            for key, value in self.word_to_index.items():
                self.index_to_word[value] = key
        
        print("\tVOCAB_SIZE {}".format(self.vocab_size()))
        logger.info("\tVOCAB_SIZE {}".format(self.vocab_size()))

    def vocab_size(self):
        return len(self.index_to_word)

    def add_word(self, word:str):
        assert word not in self.word_to_index
        self.word_to_index[word] = self.vocab_size()    # vocab_size() is the
        self.index_to_word[self.vocab_size()] = word

    @staticmethod
    def split_sentence(sentence:str)->list:
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks
    
    def encode_sentence(self, sentence, tokens=None, max_length=None)->np.ndarray:
        if max_length is None:
            max_length = self.encoding_length
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')

        encoding = [self.word_to_index['<BOS>']]
        splitted_tokens = tokens if tokens is not None else self.split_sentence(sentence)
        for word in splitted_tokens:
            encoding.append(self.word_to_index[word])   # Default Dict
        encoding.append(self.word_to_index['<EOS>'])

        if len(encoding) <= 2: return None
        if len(encoding) < max_length:
            sentence_len = len(encoding)
            encoding += [self.word_to_index['<PAD>']] * (max_length-len(encoding))  # Padding
        else: # len(encoding) >= max_length
            sentence_len = max_length           # truncated
            encoding[max_length - 1] = self.word_to_index['<EOS>']                  # Cut the length with EOS

        return np.array(encoding[:max_length]), sentence_len
    
    def decode_sentence(self, encoding:list, length=None)->str:
        sentence = []
        if length is not None:
            encoding = encoding[:length]
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.index_to_word[ix])
        return " ".join(sentence)

    def shrink(self, inst:list)->list:
        """
        :param inst:    The id inst
        :return:  Remove the potential <BOS> and <EOS>
                  If no <EOS> return empty list
        """
        if len(inst) == 0:
            return inst
        end = np.argmax(np.array(inst) == self.word_to_index['<EOS>'])     # If no <EOS>, return empty string
        if len(inst) > 1 and inst[0] == self.word_to_index['<BOS>']:
            start = 1
        else:
            start = 0
        # print(inst, start, end)
        return inst[start: end]


#######################
##     VOCAB IO      ##
#######################

def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    data = load_datasets(splits)
    for item in data:
        for instr in item['instructions']:
            count.update(Tokenizer.split_sentence(instr))
    vocab = list(start_vocab)
    for word,num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print('\tWriting vocab of size %d to %s' % (len(vocab),path))
    logger.info('\tWriting vocab of size %d to %s' % (len(vocab),path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


#######################
##     TIMER         ##
#######################

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


#######################
##     IMG FEAT      ##
#######################
import csv
csv.field_size_limit(sys.maxsize)

class ImageFeatures(object):
    NUM_VIEWS = 36
    MEAN_POOLED_DIM = 2048

    IMAGE_W = 640
    IMAGE_H = 480
    VFOV = 60

    @staticmethod
    def read_in(feature_store_path, views:int=36):
        import csv
        import base64

        print("\tStart loading the image feature")
        logger.info("\tStart loading the image feature")
        start = time.time()

        tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
        features = {}
        with open(feature_store_path, "r") as tsv_in_file:     # Open the tsv file.
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
            for item in reader:
                assert int(item['image_h']) == ImageFeatures.IMAGE_H
                assert int(item['image_w']) == ImageFeatures.IMAGE_W
                assert int(item['vfov']) == ImageFeatures.VFOV
                long_id = item['scanId'] + "_" + item['viewpointId']
                features[long_id] = np.frombuffer(
                    base64.decodestring(item['features'].encode('ascii')),
                    dtype=np.float32).reshape((views, -1))   # Feature of long_id is (36, 2048)

        print("\tFinish loading the image feature from {} in {:.4f} seconds".format(
            feature_store_path, time.time() - start))
        logger.info("\tFinish loading the image feature from {} in {:.4f} seconds".format(
            feature_store_path, time.time() - start))
        return features
    
    @staticmethod
    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId
    
    @staticmethod
    def make_angle_feat(heading, elevation, feat_size:int=128):
        import math
        # twopi = math.pi * 2
        # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
        # It will be the same
        return np.array([math.sin(heading), math.cos(heading),
                        math.sin(elevation), math.cos(elevation)],
                        dtype=np.float32).repeat(feat_size // 4)

    @staticmethod
    def build_viewpoint_loc_embedding(viewIndex, feat_size:int=128):
        """
        Position embedding:
        heading 64D + elevation 64D
        1) heading: [sin(heading) for _ in range(1, 33)] +
                    [cos(heading) for _ in range(1, 33)]
        2) elevation: [sin(elevation) for _ in range(1, 33)] +
                    [cos(elevation) for _ in range(1, 33)]
        """
        embedding = np.zeros((36, 128), np.float32)
        for absViewIndex in range(36):
            relViewIndex = (absViewIndex - viewIndex) % 12 + (absViewIndex // 12) * 12
            rel_heading = (relViewIndex % 12) * angle_inc
            rel_elevation = (relViewIndex // 12 - 1) * angle_inc
            embedding[absViewIndex, :] = \
                ImageFeatures.make_angle_feat(rel_heading, rel_elevation, feat_size)
        return embedding


# pre-compute all the 36 possible paranoram location embeddings
_static_loc_embeddings = [
    ImageFeatures.build_viewpoint_loc_embedding(viewIndex) for viewIndex in range(36)]


#######################
##     SIMULATE      ##
#######################

def _loc_distance(loc):
    return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)


def _canonical_angle(x):
    ''' Make angle in (-pi, +pi) '''
    return x - 2 * np.pi * round(x / (2 * np.pi))


def _adjust_heading(sim, heading):
    heading = (heading + 6) % 12 - 6  # minimum action to turn (e.g 11 -> -1)
    ''' Make possibly more than one heading turns '''
    for _ in range(int(abs(heading))):
        sim.makeAction(0, np.sign(heading), 0)


def _adjust_elevation(sim, elevation):
    for _ in range(int(abs(elevation))):
        ''' Make possibly more than one elevation turns '''
        sim.makeAction(0, 0, np.sign(elevation))


class M3DSimulator(object):
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    @staticmethod
    def new():
        '''
        Create a new simulator.
        '''
        # Simulator image parameters
        sim = MatterSim.Simulator()
        sim.setRenderingEnabled(False)
        sim.setDiscretizedViewingAngles(True)
        sim.setCameraResolution(M3DSimulator.WIDTH, M3DSimulator.HEIGHT)
        sim.setCameraVFOV(math.radians(M3DSimulator.VFOV))
        sim.init()

        return sim
    
    @staticmethod
    def _move_to_location(sim, nextViewpointId, absViewIndex):
        state = sim.getState()
        if state.location.viewpointId == nextViewpointId:
            return  # do nothing

        # 1. Turn to the corresponding view orientation
        _adjust_heading(sim, absViewIndex % 12 - state.viewIndex % 12)
        _adjust_elevation(sim, absViewIndex // 12 - state.viewIndex // 12)
        
        # 2. check the state
        state = sim.getState()
        assert state.viewIndex == absViewIndex

        # 3. find the next location
        a, next_loc = None, None
        for n_loc, loc in enumerate(state.navigableLocations):
            if loc.viewpointId == nextViewpointId:
                a = n_loc
                next_loc = loc
                break
        assert next_loc is not None

        # 3. Take action
        sim.makeAction(a, 0, 0)



#######################
##    LOG setting    ##
#######################

def get_main_logger(log_dir=None, model_name:str='', save_mode="dhm"):
    """ 
    Save_mode: choose in [dhm, dh, d], which means
        dhm -- day-hour-minite, dh  -- dat-hour, d   -- day
    """
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s: %(message)s", # - %(filename)s
        datefmt="%Y-%m-%d %H:%M:%S")

    # where to log
    log_dir = log_dir if log_dir is not None else\
        os.path.join(os.path.dirname(os.getcwd()), "snapshots")

    mode_str = {
        "dhm": '%Y-%m%d-%H:%M', 
        "dh" : '%Y-%m%d-%H',
        "d"  : '%Y-%m%d',
    }

    info_handler = logging.FileHandler(
        os.path.join(log_dir,
            time.strftime(mode_str[save_mode], time.localtime(time.time())) + "_" + model_name + ".log"
        ), mode='a', encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    # output to file
    logger.addHandler(info_handler)

    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(formatter)
    # # output to console
    # logger.addHandler(console_handler)

    return logger


def prettyprint(score_dict:dict):
    ''' use pretty table to print evaluation outcomes '''
    tb = pt.PrettyTable()
    tb.field_names = [' ', 'PL(↓)', 'NE(↓)', 'SR(↑)', 'OSR(↑)', 'SPL(↑)', 'nDTW(↑)', 'SDTW(↑)', 'CLS(↑)']

    for split_name, summary_dict in score_dict.items():
        raw_infomation = [
            split_name, 
            summary_dict['lengths'],
            summary_dict['nav_error'],
            summary_dict['success_rate'],
            summary_dict['oracle_rate'],
            summary_dict['spl'],
            summary_dict['ndtw'],
            summary_dict['sdtw'],
            summary_dict["cls"],
        ]
        row_content = []
        for x in raw_infomation:
            try: y = round(x, 4)
            except Exception as e: y = x
            row_content.append(y)
        tb.add_row(row_content)
    
    print(tb)
    logger.info("\n" + tb.get_string())


def pretty_json_dump(obj, fp):
    json.dump(obj, fp, sort_keys=True, indent=4, separators=(',', ':'))


#######################
##     For FGR2R     ##
#######################



#######################
##      Others       ##
#######################
import torch

def length2mask(length, device=None, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).to(device)
    return mask


#######################
##    Beam Search    ##
#######################

class FloydGraph:
    def __init__(self):
        self._dis = defaultdict(lambda :defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda :defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y:
                    if self._dis[x][k] + self._dis[k][y] < self._dis[x][y]:
                        self._dis[x][y] = self._dis[x][k] + self._dis[k][y]
                        self._dis[y][x] = self._dis[x][y]
                        self._point[x][y] = k
                        self._point[y][x] = k
        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":     # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)
