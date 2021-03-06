from raw_audio_gender_classification.config import PATH
from tqdm import tqdm
import torch.utils.data
import soundfile as sf
import pandas as pd
import numpy as np
import json
import os
import platform
from preprocessing_tools.preprocessing import preprocess
import multiprocessing
from multiprocessing import Pool
from raw_audio_gender_classification.neat.constants import *

current_os = platform.system()

sex_to_label = {'M': False, 'F': True}
label_to_sex = {False: 'M', True: 'F'}


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, subsets, length, stochastic=True, cache=True):
        """
        This class subclasses the torch Dataset object. The __getitem__ function will return a raw audio sample and it's
        label.
        :param subsets: What LibriSpeech datasets to use
        :param length: Number of audio samples to take from each file. Any files shorter than this will be ignored.
        :param stochastic: If True then we will take a random fragment from each file of sufficient length. If False we
        wil always take a fragment starting at the beginning of a file.
        """

        '''
        fixing issues Arnaud
        change long deprecated
        assert isinstance(length, (int, long)), 'Length is not an integer!'
        '''

        assert isinstance(length, int), 'Length is not an integer!'
        self.subset = subsets
        self.fragment_length = length
        self.stochastic = stochastic

        print('Initialising LibriSpeechDataset with length = {} and subsets = {}'.format(length, subsets))

        # Convert subset to list if it is a string
        # This allows to handle list of multiple subsets the same a single subset
        if isinstance(subsets, str):
            subsets = [subsets]

        # Check if we have already indexed the files
        if current_os == "Windows":
            cached_id_to_filepath_location = '/data/LibriSpeech__datasetid_to_filepath__subsets={}__length={}.json'.format(
                subsets, length)
            cached_id_to_sex_location = '/data/LibriSpeech__datasetid_to_sex__subsets={}__length={}.json'.format(
                subsets, length)
        else:
            cached_id_to_filepath_location = '/data/LibriSpeech__datasetid_to_filepath__subsets={}__length={}.json'.format(
                subsets, length)
            cached_id_to_sex_location = '/data/LibriSpeech__datasetid_to_sex__subsets={}__length={}.json'.format(
                subsets, length)
        cached_id_to_filepath_location = PATH + cached_id_to_filepath_location
        cached_id_to_sex_location = PATH + cached_id_to_sex_location

        cached_dictionaries_exist = os.path.exists(cached_id_to_filepath_location) \
                                    and os.path.exists(cached_id_to_sex_location)
        if cache and cached_dictionaries_exist:
            print('Cached indexes found.')
            with open(cached_id_to_filepath_location) as f:
                self.datasetid_to_filepath = json.load(f)

            with open(cached_id_to_sex_location) as f:
                self.datasetid_to_sex = json.load(f)

            # The dictionaries loaded from json have string type keys
            # Convert them back to integers
            '''
            fixing issues Arnaud
            iteritems no longer exist in python 3
            replace iteritems by items
            '''
            self.datasetid_to_filepath = {int(k): v for k, v in self.datasetid_to_filepath.items()}
            self.datasetid_to_sex = {int(k): v for k, v in self.datasetid_to_sex.items()}

            assert len(self.datasetid_to_filepath) == len(
                self.datasetid_to_sex), 'Cached indexes are different lengths!'

            self.n_files = len(self.datasetid_to_filepath)
            print('{} usable files found.'.format(self.n_files))

            return

        if current_os == "Windows":
            print("*********", PATH + '\\data\\LibriSpeech\\SPEAKERS.TXT')
            df = pd.read_csv(PATH + '\\data\\LibriSpeech\\SPEAKERS.TXT', skiprows=11, delimiter='|',
                             error_bad_lines=False)
        else:
            print("*********", PATH + '/data/SPEAKERS.TXT')
            df = pd.read_csv(PATH + '/data/SPEAKERS.TXT', skiprows=11, delimiter='|',
                             error_bad_lines=False)

        df.columns = [col.strip().replace(';', '').lower() for col in df.columns]
        df = df.assign(
            sex=df['sex'].apply(lambda x: x.strip()),
            subset=df['subset'].apply(lambda x: x.strip()),
            name=df['name'].apply(lambda x: x.strip()),
        )

        # Get id -> sex mapping
        librispeech_id_to_sex = df[df['subset'].isin(subsets)][['id', 'sex']].to_dict()
        self.librispeech_id_to_sex = {
            k: v for k, v in zip(librispeech_id_to_sex['id'].values(), librispeech_id_to_sex['sex'].values())}
        librispeech_id_to_name = df[df['subset'].isin(subsets)][['id', 'name']].to_dict()
        self.librispeech_id_to_name = {
            k: v for k, v in zip(librispeech_id_to_name['id'].values(), librispeech_id_to_name['name'].values())}

        datasetid = 0
        self.n_files = 0
        self.datasetid_to_filepath = {}
        self.datasetid_to_sex = {}
        self.datasetid_to_name = {}

        for s in subsets:
            print('Indexing {}...'.format(s))
            # Quick first pass to find total for tqdm bar
            subset_len = 0
            if current_os == "Windows":
                libri_path = PATH + '\\data\\LibriSpeech\\{}\\'
                for root, folders, files in os.walk(libri_path.format(s)):
                    subset_len += len([f for f in files if f.endswith('.flac')])
            else:
                libri_path = '/speechmaterials/databases/LibriSpeech/{}/'.format(s)
                print("******" + libri_path.format(s), os.path.isdir(libri_path.format(s)))

                for root, folders, files in os.walk(libri_path.format(s)):
                    subset_len += len([f for f in files if f.endswith('.flac')])

            progress_bar = tqdm(total=subset_len)
            for root, folders, files in os.walk(libri_path.format(s)):

                if len(files) == 0:
                    continue

                '''
                fixing issues Arnaud
                remove
                librispeech_id = int(root.split('/')[-2])
                '''

                for f in files:
                    # Skip non-sound files
                    if not f.endswith('.flac'):
                        continue

                    '''
                    fixing issues Arnaud
                    adding
                    '''
                    librispeech_id = int(files[0].split('-')[0])
                    if librispeech_id not in self.librispeech_id_to_sex:
                        continue
                    progress_bar.update(1)

                    # Skip short files
                    instance, samplerate = sf.read(os.path.join(root, f))
                    if len(instance) <= self.fragment_length:
                        continue
                    # TODO fixing issues with librispeech_id or whatever
                    self.datasetid_to_filepath[datasetid] = os.path.abspath(os.path.join(root, f))
                    self.datasetid_to_sex[datasetid] = self.librispeech_id_to_sex[librispeech_id]
                    self.datasetid_to_name[datasetid] = self.librispeech_id_to_name[librispeech_id]
                    datasetid += 1
                    self.n_files += 1

            progress_bar.close()
        print('Finished indexing data. {} usable files found.'.format(self.n_files))

        # Save relevant dictionaries to json in order to re-use them layer
        # The indexing takes a few minutes each time and would be nice to just perform this calculation once
        with open(cached_id_to_filepath_location, 'w') as f:
            json.dump(self.datasetid_to_filepath, f)

        with open(cached_id_to_sex_location, 'w') as f:
            json.dump(self.datasetid_to_sex, f)

    def __getitem__(self, index):
        instance, samplerate = sf.read(self.datasetid_to_filepath[index])
        # Choose a random sample of the file
        if self.stochastic:
            fragment_start_index = np.random.randint(0, len(instance) - self.fragment_length)
        else:
            fragment_start_index = 0
        instance = instance[fragment_start_index:fragment_start_index + self.fragment_length]
        sex = self.datasetid_to_sex[index]
        return instance, sex_to_label[sex]

    def __len__(self):
        return self.n_files


def preprocess_function(s):
    return torch.from_numpy(preprocess(s, option=OPTION, bins=BINS, sr=16000, win_length=WIN_LENGTH, hop_length=HOP_LENGTH))


class PreprocessedLibriSpeechDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: LibriSpeechDataset):
        self.len = len(dataset)
        self.X = torch.empty(self.len, 16000 * 3 // HOP_LENGTH + 1, BINS, dtype=torch.float32)
        self.t = torch.empty(self.len, dtype=torch.float32)

        with Pool(multiprocessing.cpu_count()) as pool:
            jobs = []

            for i in tqdm(range(self.len)):
                x, t = dataset[i]
                jobs.append(pool.apply_async(preprocess_function,
                                             (x.astype(np.float32),)))
                self.t[i] = float(t)

            for i, job in enumerate(jobs):
                self.X[i] = job.get(timeout=None)

    def __getitem__(self, index):
        return self.X[index], self.t[index]

    def __len__(self):
        return self.len
