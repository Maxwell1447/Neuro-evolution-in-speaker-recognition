"""

    adapted from https://github.com/nesl/asvspoof2019
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
    
"""


from tqdm import tqdm
import collections
import os
import soundfile as sf
from torch.utils.data import Dataset
import platform
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_os = platform.system()

DATA_ROOT = 'data'

ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

class ASVDataset_stats(Dataset):
    """ Utility class to load  train/dev datatsets """
    def __init__(self, length, transform=None,
        is_train=True, is_logical=True, is_eval=False):
        data_root = DATA_ROOT
        if is_logical:
            track = 'LA'
        else:
            track = 'PA'
        if is_eval:
            data_root = os.path.join('eval_data', data_root)
        self.fragment_length = length
        self.track = track
        self.is_train = is_train
        self.is_logical = is_logical
        self.prefix = 'ASVspoof2019_{}'.format(track)
        v1_suffix = ''
        if is_eval and track == 'PA':
            v1_suffix = '_v1'
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
        self.sysid_dict_inv = {v:k for k, v in self.sysid_dict.items()}
        self.data_root = data_root
        self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
        self.protocols_fname = 'eval.trl' if is_eval else 'train.trn' if is_train else 'dev.trl'
        self.protocols_dir = os.path.join(self.data_root,
            '{}_protocols/'.format(self.prefix))
        self.files_dir = os.path.join(self.data_root, '{}_{}'.format(
            self.prefix, self.dset_name )+v1_suffix, 'flac')
        if current_os == "Windows":
            self.protocols_fname = 'data\\{}\\ASVspoof2019_{}_cm_protocols\\ASVspoof2019.{}.cm.{}.txt'.format(track, track, track, self.protocols_fname)
        else:
            self.protocols_fname = 'data/{}/ASVspoof2019_{}_cm_protocols/ASVspoof2019.{}.cm.{}.txt'.format(track, track, track, self.protocols_fname)
        self.cache_fname = 'cache_{}{}_{}.npy'.format(self.dset_name,'',track)
        
        self.files_meta = self.parse_protocols_file(self.protocols_fname)
        # tqdm bar
        data = list(map(self.read_file, tqdm(self.files_meta)))
        self.data_x, self.data_y = map(list, zip(*data))
        self.length = len(self.data_x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y 
        # to add meta data
        # self.files_meta[idx]

    def read_file(self, meta):
        '''
        return length and class for one audio file
        '''
        if current_os == "Windows":
            if self.is_train:
                tmp_path = meta.path[:5] + self.track + "\\" + meta.path[5:]
            elif self.is_eval:
                tmp_path = meta.path[10:15] + self.track + "\\" + meta.path[15:]
            else:
                tmp_path = meta.path[:5] + self.track + "\\" + meta.path[5:]
        else:
            if self.is_train:
                tmp_path = meta.path[:5] + self.track + "/" + meta.path[5:]
            elif self.is_eval:
                tmp_path = meta.path[10:15] + self.track + "/" + meta.path[15:]
            else:
                tmp_path = meta.path[:5] + self.track + "/" + meta.path[5:]
        data_x, sample_rate = sf.read(tmp_path)
        data_y = meta.key
        return data_x.size, float(data_y)

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[4] == 'bonafide'))

    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()
        files_meta = map(self._parse_line, lines)
        return list(files_meta)

if __name__ == '__main__':
    train_meta_data_loader = ASVDataset_stats(48000, is_train=True)
    eval_meta_data_loader = ASVDataset_stats(48000, is_train=False, is_eval=True)
    dev_meta_data_loader = ASVDataset_stats(48000, is_train=False, is_eval=False)
    
    
    train_audio_length = np.array(train_meta_data_loader.data_x) / 16000
    train_outputs = np.array(train_meta_data_loader.data_y)
    bonafide_train = train_outputs[train_outputs==1].size
    spoofed_train = train_outputs[train_outputs==0].size
    
    
    eval_audio_length = np.array(eval_meta_data_loader.data_x) / 16000
    eval_outputs = np.array(eval_meta_data_loader.data_y)
    bonafide_eval = eval_outputs[eval_outputs==1].size
    spoofed_eval = eval_outputs[eval_outputs==0].size
    
    dev_audio_length = np.array(dev_meta_data_loader.data_x) / 16000
    dev_outputs = np.array(dev_meta_data_loader.data_y)
    bonafide_dev = dev_outputs[dev_outputs==1].size
    spoofed_dev = dev_outputs[dev_outputs==0].size
    
    #### outputs ####
    outputs = pd.DataFrame({'bonafide': [bonafide_train, bonafide_eval, bonafide_dev], 'spoofed': [spoofed_train, spoofed_eval, spoofed_dev]},index=['train', 'eval','dev'])
    plt.figure(figsize = (14,7))
    plt.title("Outputs ASVDataset 2019 Logical")
    ax = sns.heatmap( data = outputs, annot=True, fmt="d", center = None, linewidths=.5, cmap="RdBu")
    # Unfortunately matplotlib 3.1.1 broke seaborn heatmaps; to solve the problem
    ax.set_ylim(top=0, bottom=outputs.index.size)
    
    #### length inputs ####
    sns.set(style="darkgrid")
    plt.figure(figsize = (14,7))
    df_train_length = pd.Series(train_audio_length, name="train audio files length")
    ax2 = sns.distplot(df_train_length, norm_hist=False)
    
    
    plt.figure(figsize = (14,7))
    df_eval_length = pd.Series(eval_audio_length, name="eval audio files length")
    ax3 = sns.distplot(df_eval_length, norm_hist=False)
    
    plt.figure(figsize = (14,7))
    df_dev_length = pd.Series(dev_audio_length, name="dev audio files length")
    ax4 = sns.distplot(df_dev_length, norm_hist=False)

    