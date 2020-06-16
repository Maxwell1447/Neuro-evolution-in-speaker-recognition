"""

    adapted from https://github.com/nesl/asvspoof2019
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
    
"""


from tqdm import tqdm
import torch
import collections
import os
import soundfile as sf
from torch.utils.data import Dataset
import numpy as np
import platform

current_os = platform.system()

DATA_ROOT = 'data'

ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

class ASVDatasetshort(Dataset):
    """ Utility class to load  train/dev datatsets """
    def __init__(self, length, nb_samples=10000,
        sample_size=None,
        save_cache=False, index_list = None):
        data_root = DATA_ROOT
        track = 'LA'
        self.track ='LA'
        self.fragment_length = length
        self.prefix = 'ASVspoof2019_{}'.format(track)
        self.nb_samples = nb_samples
        self.index_list = index_list
        v1_suffix = ''
        self.sysid_dict = {
            'human': 0, # bonafide speech
            'A01': 1, # Wavenet vocoder
            'A02': 2, # Conventional vocoder WORLD
            'A03': 3, # Conventional vocoder MERLIN
            'A04': 4, # Unit selection system MaryTTS
            'A05': 5, # Voice conversion using neural networks
            'A06': 6, # transform function-based voice conversion
        }
        self.sysid_dict_inv = {v:k for k, v in self.sysid_dict.items()}
        self.data_root = data_root
        self.dset_name = 'train'
        self.protocols_fname = 'train_short.trn'
        self.protocols_dir = os.path.join(self.data_root,
            '{}_protocols/'.format(self.prefix))
        self.files_dir = os.path.join(self.data_root, '{}_{}'.format(
            self.prefix, self.dset_name )+v1_suffix, 'flac')
        if current_os == "Windows":
            self.protocols_fname = 'data\\{}\\ASVspoof2019_{}_cm_protocols\\ASVspoof2019.{}.cm.{}.txt'.format(track, track, track, self.protocols_fname)
        else:
            self.protocols_fname = 'data/{}/ASVspoof2019_{}_cm_protocols/ASVspoof2019.{}.cm.{}.txt'.format(track, track, track, self.protocols_fname)
        self.cache_fname = 'cache_{}{}_{}.npy'.format(self.dset_name,'',track)
        if os.path.exists(self.cache_fname):
            self.data_x, self.data_y, self.data_sysid, self.files_meta = torch.load(self.cache_fname)
            print('Dataset loaded from cache ', self.cache_fname)
        else:
            self.files_meta = self.parse_protocols_file(self.protocols_fname)
            # tqdm bar
            data = list(map(self.read_file, tqdm(self.files_meta)))
            self.data_x, self.data_y = map(list, zip(*data))
            # to add meta data    
            # self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
            if save_cache:
                torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta), self.cache_fname)
                print('Dataset saved to cache ', self.cache_fname)
        if sample_size:
            select_idx = np.random.choice(len(self.files_meta), size=(sample_size,), replace=True).astype(np.int32)
            self.files_meta= [self.files_meta[x] for x in select_idx]
            self.data_x = [self.data_x[x] for x in select_idx]
            self.data_y = [self.data_y[x] for x in select_idx]
            #self.data_sysid = [self.data_sysid[x] for x in select_idx]
        self.length = len(self.data_x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y, self.files_meta[idx].sys_id
        # to all of the meta data
        # self.files_meta[idx]

    def read_file(self, meta):
        if current_os == "Windows":
            tmp_path = meta.path[:5] + self.track + "\\" + meta.path[5:]
        else:
            tmp_path = meta.path[:5] + self.track + "/" + meta.path[5:]
        data_x, sample_rate = sf.read(tmp_path)
        data_y = meta.key
        # to make all data to have the same length
        if self.fragment_length:
            if data_x.size < self.fragment_length:
                nb_iter =  self.fragment_length // data_x.size + 1 
                data_x_copy = data_x[:]
                for _ in range(nb_iter):
                    data_x = np.concatenate((data_x, data_x_copy))
            return data_x[:self.fragment_length], float(data_y)
        else:
            return data_x, float(data_y)
        # to add meta data    
        # meta.sys_id

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[2]],
            key=int(tokens[3] == 'human'))

    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()
        files_meta = map(self._parse_line, lines)
        meta_files_list = list(files_meta)
        if self.index_list:
            return [meta_files_list[i] for i in self.index_list ]
        return meta_files_list[:self.nb_samples]

if __name__ == '__main__':
    train_loader = ASVDatasetshort(48000, nb_samples=10)