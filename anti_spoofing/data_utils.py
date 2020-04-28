"""
    adapted from https://github.com/nesl/asvspoof2019
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
    
"""


import torch
import collections
import os
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import Parallel, delayed
import h5py

DATA_ROOT = 'data'

ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

class ASVDataset(Dataset):
    """ Utility class to load  train/dev datatsets """
    def __init__(self, transform=None, nb_samples=10000,
        is_train=True, sample_size=None, 
        is_logical=True, feature_name=None, is_eval=False,
        eval_part=0, save_cache=False):
        data_root = DATA_ROOT
        if is_logical:
            track = 'LA'
        else:
            track = 'PA'
        if is_eval:
            data_root = os.path.join('eval_data', data_root)
        self.track = track
        self.is_train = is_train
        self.is_logical = is_logical
        self.prefix = 'ASVspoof2019_{}'.format(track)
        self.nb_samples = nb_samples
        v1_suffix = ''
        if is_eval and track == 'PA':
            v1_suffix='_v1'
        self.sysid_dict = {
            '-': 0,  # bonafide speech
            'A01': 1, # Wavenet vocoder
            'A02': 2, # Conventional vocoder WORLD
            'A03': 3, # Conventional vocoder MERLIN
            'A04': 4, # Unit selection system MaryTTS
            'A05': 5, # Voice conversion using neural networks
            'A06': 6, # transform function-based voice conversion
            # For PA:
            'A07':7,
            'A08':8,
            'A09':9,
            'A10':10,
            'A11':11,
            'A12':12,
            'A13':13,
            'A14':14,
            'A15':15,
            'A16':16,
            'A17':17,
            'A18':18,
            'A19':19
        }
        self.is_eval = is_eval
        self.sysid_dict_inv = {v:k for k,v in self.sysid_dict.items()}
        self.data_root = data_root
        self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
        self.protocols_fname = 'eval.trl'.format(eval_part) if is_eval else 'train.trn' if is_train else 'dev.trl'
        self.protocols_dir = os.path.join(self.data_root,
            '{}_protocols/'.format(self.prefix))
        self.files_dir = os.path.join(self.data_root, '{}_{}'.format(
            self.prefix, self.dset_name )+v1_suffix, 'flac')
        self.protocols_fname = 'data\\{}\\ASVspoof2019_{}_cm_protocols\\ASVspoof2019.{}.cm.{}.txt'.format(track, track, track, self.protocols_fname)
        self.cache_fname = 'cache_{}{}_{}.npy'.format(self.dset_name,'',track)
        self.transform = transform
        if os.path.exists(self.cache_fname):
            self.data_x, self.data_y, self.data_sysid, self.files_meta = torch.load(self.cache_fname)
            print('Dataset loaded from cache ', self.cache_fname)
        else:
            self.files_meta = self.parse_protocols_file(self.protocols_fname)
            data = list(map(self.read_file, self.files_meta))
            self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
            if save_cache:
                torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta), self.cache_fname)
                print('Dataset saved to cache ', self.cache_fname)
        if sample_size:
            select_idx = np.random.choice(len(self.files_meta), size=(sample_size,), replace=True).astype(np.int32)
            self.files_meta= [self.files_meta[x] for x in select_idx]
            self.data_x = [self.data_x[x] for x in select_idx]
            self.data_y = [self.data_y[x] for x in select_idx]
            self.data_sysid = [self.data_sysid[x] for x in select_idx]
        self.length = len(self.data_x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y, self.files_meta[idx]

    def read_file(self, meta):
        print(meta.path)
        if self.is_train:
            tmp_path = meta.path[:5] + self.track + "\\" + meta.path[5:]
        elif self.is_eval:
            tmp_path = meta.path[10:15] + self.track + "\\" + meta.path[15:]
        else:
            tmp_path = meta.path[:5] + self.track + "\\" + meta.path[5:]
        data_x, sample_rate = sf.read(tmp_path)
        data_y = meta.key
        return data_x, float(data_y), meta.sys_id

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[4] == 'bonafide'))

    def parse_protocols_file(self, protocols_fname):
        print(protocols_fname)
        lines = open(protocols_fname).readlines()
        files_meta = map(self._parse_line, lines)
        return list(files_meta)[:self.nb_samples]

if __name__ == '__main__':
   #train_loader = ASVDataset(DATA_ROOT, is_train=True)
   testset = ASVDataset(DATA_ROOT, is_train=False, is_eval=True)