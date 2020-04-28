from tqdm import tqdm
import torch.utils.data
import soundfile as sf
import pandas as pd
import numpy as np
import json
import os
import platform
import collections

'''
does not work
attempt to adapt LibriSpeech data loader to ASV2019 dataset
'''

current_os = platform.system()
PATH = os.path.dirname(os.path.realpath(__file__))


ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

class ASVDataset(torch.utils.data.Dataset):
    def __init__(self, track, dataset_type, length, stochastic=True, cache=True):
        """
        This class subclasses the torch Dataset object. The __getitem__ function will return a raw audio sample and it's
        label.
        :param track: What ASV datasets to use PA or LA 
        :param dataset_type: train / eval / dev
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
        self.track = track
        self.fragment_length = length
        self.stochastic = stochastic
        self.dataset_type = dataset_type

        print('Initialising ASVDataset {} with length = {}'.format(track, length))

        # Convert subset to list if it is a string
        # This allows to handle list of multiple subsets the same a single subset

        # Check if we have already indexed the files
        if current_os == "Windows":
            cached_id_to_filepath_location = '/data/ASV_dataset__datasetid_to_filepath__track={}_{}_length={}.json'.format(
                track, dataset_type, length)
            cached_id_to_type_location = '/data/ASV_dataset__datasetid_to_type__track={}_{}_length={}.json'.format(
                track, dataset_type, length)
        else:
            cached_id_to_filepath_location = '/data/ASV_dataset__datasetid_to_filepath__track={}_{}_length={}.json'.format(
                track, dataset_type, length)
            cached_id_to_type_location = '/data/ASV_dataset__datasetid_to_type__track={}_{}_length={}.json'.format(
                track, dataset_type, length)
        cached_id_to_filepath_location = PATH + cached_id_to_filepath_location
        cached_id_to_type_location = PATH + cached_id_to_type_location

        cached_dictionaries_exist = os.path.exists(cached_id_to_filepath_location) \
                                    and os.path.exists(cached_id_to_type_location)
        if cache and cached_dictionaries_exist:
            print('Cached indexes found.')
            with open(cached_id_to_filepath_location) as f:
                self.datasetid_to_filepath = json.load(f)

            with open(cached_id_to_type_location) as f:
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
            print("*********", PATH + '\\data\\{}\\ASVspoof2019_PA_cm_protocols\\ASVspoof2019.{}.cm.{}.txt'.format(track, track, dataset_type))
            df = data = list(map(self.read_file, self.files_meta))
            self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
        else:
            print("*********", PATH + '/data/{}/ASVspoof2019_PA_cm_protocols/ASVspoof2019.{}.cm.{}.txt'.format(track, track, dataset_type))
            df = pd.read_csv(PATH + '/data/SPEAKERS.TXT', skiprows=11, delimiter='|',
                             error_bad_lines=False)

        df.columns = [col.strip().replace(';', '').lower() for col in df.columns]
        df = df.assign(
            sex=df['sex'].apply(lambda x: x.strip()),
            subset=df['subset'].apply(lambda x: x.strip()),
            name=df['name'].apply(lambda x: x.strip()),
        )

        # Get id
        

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
                print("******"+libri_path.format(s), os.path.isdir(libri_path.format(s)))

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

        with open(cached_id_to_type_location, 'w') as f:
            json.dump(self.datasetid_to_sex, f)
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y, self.files_meta[idx]

    def read_file(self, meta):
        print(meta.path)
        tmp_path = meta.path[:5] + self.track + "\\" + meta.path[5:]
        data_x, sample_rate = sf.read(tmp_path)
        data_y = meta.key
        return data_x, float(data_y), meta.sys_id

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        if self.is_eval:
            return ASVFile(speaker_id='',
                file_name=tokens[0],
                path=os.path.join(self.files_dir, tokens[0] + '.flac'),
                sys_id=0,
                key=0)
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[4] == 'bonafide'))

    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()
        files_meta = map(self._parse_line, lines)
        return list(files_meta)

    def read_matlab_cache(self, filepath):
        f = h5py.File(filepath, 'r')
        # filename_index = f["filename"]
        # filename = []
        data_x_index = f["data_x"]
        sys_id_index = f["sys_id"]
        data_x = []
        data_y = f["data_y"][0]
        sys_id = []
        for i in range(0, data_x_index.shape[1]):
            idx = data_x_index[0][i]  # data_x
            temp = f[idx]
            data_x.append(np.array(temp).transpose())
            # idx = filename_index[0][i]  # filename
            # temp = list(f[idx])
            # temp_name = [chr(x[0]) for x in temp]
            # filename.append(''.join(temp_name))
            idx = sys_id_index[0][i]  # sys_id
            temp = f[idx]
            sys_id.append(int(list(temp)[0][0]))
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        return data_x.astype(np.float32), data_y.astype(np.int64), sys_id
