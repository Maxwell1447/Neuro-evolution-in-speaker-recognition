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
import random
import librosa
from spafe.features.lfcc import lfcc

from anti_spoofing.utils_ASV import whiten
from anti_spoofing.mfcc import mfcc


ASVFile = collections.namedtuple('ASVFile',
                                 ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])


class ASVDatasetshort(Dataset):
    """
    Utility class to load  train short data set
    """
    def __init__(self, length, nb_samples=2538, random_samples=False,
                 sample_size=None,
                 save_cache=False, custom_path="./data", index_list=None,
                 do_standardize=False, do_mfcc=False, do_chroma_cqt=False, do_chroma_stft=False, do_self_mfcc=False,
                 do_lfcc=False, n_fft=2048, do_mrf=False, metadata=True):
        """
        :param length: int
        Length of the audio files in number of elements in a numpy array format.
        Number of elements times the sampling rate is equal the length in seconds of the audio file.
        It will set the audio files to the correspond length by removing the end or adding duplicate parts
        of the file.
        If set to None, it will return the audio files without changing their length.
        :param nb_samples: int
        Number of files to use. If greater than the actual number of files, nb_samples will be set to this value.
        :param random_samples: bool
        If true then the files will be chosen randomly (shuffled if all the files are selected)
        :param sample_size: int
        Number of files to use. Difference between this one and nb_samples, is that, nb_samples will only load the
        correct number of files required, whereas sample_size will load all files from the folder considered and then
        randomly choose which files to keep.
        :param save_cache: bool
        If True, will save the cache with torch.
        :param custom_path: str
        directory when ASV data in a specific folder
        :param index_list: list
        If set to a non empty list, will only use the audio files whose index is in the list
        :param do_standardize: bool
        If True will standardize the audio files.
        :param do_mfcc: bool
        If True will return the Mel-frequency cepstral coefficients (mfcc) of the audio files
        and not the raw audio files.
        :param do_chroma_cqt: bool
        If True will return the Constant-Q chromagram (cqt) of the audio files
        and not the raw audio files.
        :param do_chroma_stft: bool
        If True will return the chromagram, Short-time Fourier transform (stft), from the audio files
        and not the raw audio files.
        :param do_self_mfcc: bool
        If True will return the Mel-frequency cepstral coefficients (mfcc) of the audio files
        and not the raw audio files. This version does not use librosa.
        :param do_lfcc: bool
        If True, compute the linear-frequency cepstral coefﬁcients (GFCC features) from the audio signal.
        :param n_fft: int or list of int
        length of the FFT window
        :param do_mrf: bool
        If yes, will use Multi-Resolution Feature Maps
        """
        data_root = custom_path
        track = 'LA'
        self.track = 'LA'
        self.metadata = metadata
        self.fragment_length = length
        self.prefix = 'ASVspoof2019_{}'.format(track)
        self.nb_samples = nb_samples
        self.random_samples = random_samples
        self.index_list = index_list
        self.standardize = do_standardize
        self.mfcc = do_mfcc
        self.chroma_cqt = do_chroma_cqt
        self.chroma_stft = do_chroma_stft
        self.m_mfcc = do_self_mfcc
        self.n_fft = n_fft
        self.mrf = do_mrf
        self.lfcc = do_lfcc
        if self.fragment_length and (self.chroma_stft or self.mfcc or self.chroma_cqt):
            raise ValueError("You cannot specify a length if you are using pre-processing functions")
        v1_suffix = ''
        self.sysid_dict = {
            'human': 0,  # bonafide speech
            'A01': 1,  # Wavenet vocoder
            'A02': 2,  # Conventional vocoder WORLD
            'A03': 3,  # Conventional vocoder MERLIN
            'A04': 4,  # Unit selection system MaryTTS
            'A05': 5,  # Voice conversion using neural networks
            'A06': 6,  # transform function-based voice conversion
        }
        self.sysid_dict_inv = {v: k for k, v in self.sysid_dict.items()}
        self.data_root = data_root
        self.dset_name = 'train'
        self.protocols_fname = 'train_short.trn'
        self.protocols_dir = os.path.join(self.data_root,
                                          '{}_protocols/'.format(self.prefix))
        self.files_dir = os.path.join(self.data_root, track, '{}_{}'.format(
            self.prefix, self.dset_name) + v1_suffix, 'flac')

        self.protocols_fname = os.path.join(custom_path, track, 'ASVspoof2019_{}_cm_protocols'.format(track),
                                            'ASVspoof2019.{}.cm.{}.txt'.format(track, self.protocols_fname))

        assert os.path.isfile(self.protocols_fname)

        self.cache_fname = 'cache_{}{}_{}.npy'.format(self.dset_name, '', track)
        if os.path.exists(self.cache_fname):
            self.data_x, self.data_y, self.data_sysid, self.files_meta = torch.load(self.cache_fname)
            print('Dataset loaded from cache ', self.cache_fname)
        else:
            self.files_meta = self.parse_protocols_file(self.protocols_fname)
            # tqdm progress for loading files
            data = list(map(self.read_file, tqdm(self.files_meta)))
            self.data_x, self.data_y = map(list, zip(*data))
            # to add meta data    
            # self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
            if save_cache:
                torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta), self.cache_fname)
                print('Dataset saved to cache ', self.cache_fname)
        if sample_size:
            select_idx = np.random.choice(len(self.files_meta), size=(sample_size,), replace=True).astype(np.int32)
            if metadata:
                self.files_meta = [self.files_meta[x] for x in select_idx]
            self.data_x = [self.data_x[x] for x in select_idx]
            self.data_y = [self.data_y[x] for x in select_idx]
            # self.data_sysid = [self.data_sysid[x] for x in select_idx]
        self.length = len(self.data_x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        if self.metadata:
            return x, y, self.files_meta[idx].sys_id
        else:
            return x, y
        # to all of the meta data
        # self.files_meta[idx]

    def read_file(self, meta):

        tmp_path = meta.path

        data_x, sample_rate = sf.read(tmp_path)
        data_y = meta.key
        if self.mfcc:
            data_x = librosa.feature.mfcc(y=data_x, sr=sample_rate, n_mfcc=24, n_fft=self.n_fft)
        if self.chroma_cqt:
            data_x = librosa.feature.chroma_cqt(y=data_x, sr=sample_rate, n_chroma=24)
        if self.chroma_stft:
            data_x = librosa.feature.chroma_stft(y=data_x, sr=sample_rate, n_chroma=24, n_fft=self.n_fft)
        if self.m_mfcc:
            data_x = mfcc(data_x, num_cep=24, nfft=self.n_fft)
        if self.lfcc:
            data_x = lfcc(data_x, fs=sample_rate, num_ceps=20, pre_emph=0, pre_emph_coeff=0.97, win_len=0.030,
                          win_hop=0.015, win_type="hamming", nfilts=70, nfft=1024, low_freq=0, high_freq=8000,
                          scale="constant", dct_type=2, use_energy=False, lifter=22, normalize=0)
        if self.standardize:
            data_x = whiten(data_x)
        if self.mrf:
            copy_data_x = data_x[:]
            data_x = librosa.feature.chroma_stft(y=data_x, sr=sample_rate, n_chroma=24, n_fft=self.n_fft[0])
            if self.standardize:
                data_x = whiten(data_x)
            for index_n_fft in range(1, len(self.n_fft)):
                fft_data_x = librosa.feature.chroma_stft(y=copy_data_x, sr=sample_rate,
                                                         n_chroma=24, n_fft=self.n_fft[index_n_fft])
                if self.standardize:
                    fft_data_x = whiten(fft_data_x)
                data_x = np.concatenate((data_x, fft_data_x))

        # to make all data to have the same length
        if self.fragment_length:
            if data_x.size < self.fragment_length:
                nb_iter = self.fragment_length // data_x.size + 1
                data_x = np.tile(data_x, nb_iter)

            begin = np.random.randint(0, data_x.size - self.fragment_length)
            return data_x[begin: begin + self.fragment_length], float(data_y)
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
            return [meta_files_list[i] for i in self.index_list]
        if self.random_samples:
            self.nb_samples = np.min([self.nb_samples, len(meta_files_list)])
            random_index = random.sample(range(0, len(meta_files_list)), self.nb_samples)
            return [meta_files_list[i] for i in random_index]
        return meta_files_list[:self.nb_samples]


if __name__ == '__main__':
    train_loader = ASVDatasetshort(length=None, nb_samples=10, do_lfcc=True, do_standardize=True)
