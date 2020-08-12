from anti_spoofing.data_utils import ASVDataset
from anti_spoofing.constants import *
import torch
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

from anti_spoofing.matlab.mat_loader import CQCCDataset
from preprocessing_tools.preprocessing import preprocess
import os
from torch.utils.data.dataloader import DataLoader


def preprocess_function(s):
    return preprocess(s, option=OPTION, bins=BINS, sr=16000)


class PreprocessedASVDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: ASVDataset, multi_proc=True, balanced=True):
        self.len = len(dataset)
        self.balanced = balanced
        self.X = torch.empty(self.len, 16000 * 3 // 512 + 1, BINS, dtype=torch.float32)
        self.t = torch.empty(self.len, dtype=torch.float32)
        self.data_idx_s = None
        self.data_idx_b = None
        if dataset.metadata:
            self.meta = torch.empty(self.len, dtype=torch.int64)
        else:
            self.meta = None

        if multi_proc:
            # torch.multiprocessing.set_sharing_strategy('file_descriptor')

            with Pool(multiprocessing.cpu_count()) as pool:
                jobs = []

                for i in tqdm(range(self.len)):
                    if dataset.metadata:
                        x, t, meta = dataset[i]
                    else:
                        x, t = dataset[i]
                    jobs.append(pool.apply_async(preprocess_function, [x.astype(np.float32)]))
                    self.t[i] = float(t)
                    if dataset.metadata:
                        self.meta[i] = int(meta)

                for i, job in tqdm(enumerate(jobs), total=self.len):
                    self.X[i] = torch.from_numpy(job.get(timeout=30))
        else:
            for i in tqdm(range(self.len)):
                if dataset.metadata:
                    x, t, meta = dataset[i]
                    self.meta[i] = int(meta)
                else:
                    x, t = dataset[i]
                self.X[i] = preprocess_function(x.astype(np.float32))
                self.t[i] = float(t)

        self.set_balance(balanced)

    def __getitem__(self, index):
        if self.balanced:
            if index < self.len // 2:
                idx_s = self.data_idx_s[index]
                assert self.t[idx_s] < 0.5
                return self.get_data(idx_s)
            else:
                idx_b = self.data_idx_b[index % len(self.data_idx_b)]
                assert self.t[idx_b] > 0.5
                return self.get_data(idx_b)
        else:
            return self.get_data(index)

    def get_data(self, index):
        if self.meta is not None:
            return self.X[index], self.t[index], self.meta[index]
        else:
            return self.X[index], self.t[index]

    def __len__(self):
        return self.len

    def set_balance(self, value: bool):
        self.balanced = value
        if self.balanced:
            idxs = torch.arange(len(self.t))
            self.data_idx_s = idxs[self.t < 0.5]  # N
            self.data_idx_b = idxs[self.t > 0.5]  # n << N
            self.len = 2 * len(self.data_idx_s)
        else:
            self.len = len(self.t)


def load_data(batch_size=50, batch_size_test=1, length=3 * 16000, num_train=10000, num_test=10000, custom_path='./data',
              multi_proc=True, balanced=True):
    """
    loads the data and puts it in PyTorch DataLoader.
    Librispeech uses Index caching to access the data more rapidly.

    If a data loader has not been saved already,
    a data loader is created, then saved for train and test sets.
    """

    train_loader = load_single_data(batch_size=batch_size, length=length, num_data=num_train, data_type="train",
                                    custom_path=custom_path, multi_proc=multi_proc, balanced=balanced)
    test_loader = load_single_data(batch_size=batch_size_test, length=length, num_data=num_test, data_type="test",
                                   custom_path=custom_path, multi_proc=multi_proc, balanced=balanced)

    return train_loader, test_loader


def load_data_cqcc(batch_size=50, batch_size_test=1, num_train=1000, num_test=1000, balanced=False):
    """
    loads the data and puts it in PyTorch DataLoader.
    Librispeech uses Index caching to access the data more rapidly.

    If a data loader has not been saved already,
    a data loader is created, then saved for train and test sets.
    """

    train_dataloader = load_single_data_cqcc(batch_size=batch_size, num_data=num_train,
                                             balanced=balanced, data_type="train")
    dev_dataloader = load_single_data_cqcc(batch_size=batch_size_test, num_data=num_test,
                                           balanced=balanced, data_type="test")

    return train_dataloader, dev_dataloader


def load_metadata(batch_size=50, batch_size_test=1, length=3 * 16000, num_train=10000, num_test=10000,
                  custom_path='./data',
                  multi_proc=True):
    """
    loads the data and the metadata and puts it in PyTorch DataLoader.

    If a data loader has not been saved already,
    a data loader is created, then saved for train and test sets.
    """

    train_loader = load_single_metadata(batch_size=batch_size, length=length, num_data=num_train, data_type="train",
                                        custom_path=custom_path, multi_proc=multi_proc)
    test_loader = load_single_metadata(batch_size=batch_size_test, length=length, num_data=num_test, data_type="test",
                                       custom_path=custom_path, multi_proc=multi_proc)

    return train_loader, test_loader


def load_single_data(batch_size=50, length=3 * 16000, num_data=10000, data_type="train", custom_path="./data",
                     multi_proc=True, balanced=True):
    option = OPTION

    is_train = data_type == "train"

    local_dir = os.path.dirname(__file__)

    if os.path.exists(os.path.join(local_dir,
                                   "data/preprocessed/{}_{}_{}.torch".format(data_type, option, num_data))):
        data = torch.load(os.path.join(local_dir,
                                       "data/preprocessed/{}_{}_{}.torch".format(data_type, option, num_data)))
        data.set_balance(balanced and is_train)
        dataloader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=is_train, drop_last=is_train)
        return dataloader

    if not os.path.isdir(os.path.join(local_dir, 'data/preprocessed')):
        os.makedirs(os.path.join(local_dir, 'data/preprocessed'))

    if data_type == "train":
        data = ASVDataset(length=length, nb_samples=num_data, random_samples=True, metadata=False,
                          custom_path=custom_path)
    else:
        data = ASVDataset(length=length, is_train=False, is_eval=False, nb_samples=num_data, random_samples=True,
                          metadata=False, custom_path=custom_path)

    print("preprocessing_tools {} set".format(data_type))
    pp_data = PreprocessedASVDataset(data, multi_proc=multi_proc, balanced=is_train and balanced)
    torch.save(pp_data, os.path.join(local_dir,
                                     "data/preprocessed/{}_{}_{}.torch".format(data_type, option, num_data)))
    dataloader = DataLoader(pp_data, batch_size=batch_size, num_workers=4, shuffle=is_train, drop_last=is_train)

    return dataloader


def load_single_data_cqcc(batch_size=50, num_data=1000, balanced=False, data_type="train"):
    local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "anti_spoofing")

    is_train = data_type == "train"

    if os.path.exists(
            os.path.join(local_dir, "data", "preprocessed", "{}_cqcc_{}_{}_{}_{}_{}"
                    .format(data_type, B, d, cf, ZsdD, balanced))):
        print("{} data found locally".format(data_type))
        cqcc_data = torch.load("./data/preprocessed/{}_cqcc_{}_{}_{}_{}_{}"
                               .format(data_type, B, d, cf, ZsdD, balanced))
        dataloader = torch.utils.data.DataLoader(cqcc_data, batch_size=batch_size,
                                                 num_workers=0, shuffle=is_train, drop_last=True)
        return dataloader

    if not os.path.isdir('./data/preprocessed'):
        local_dir = os.path.dirname(__file__)
        os.makedirs(os.path.join(local_dir, 'data/preprocessed'))

    print("loading {} .mat files to PyTorch...".format(data_type))

    cqcc_data_type = data_type if data_type == "train" else "dev"

    cqcc_data = CQCCDataset(params_id="{}_{}_{}_{}_{}".format(cqcc_data_type, B, d, cf, ZsdD),
                            n_files=num_data, balanced=is_train and balanced)

    torch.save(cqcc_data,
               os.path.join(local_dir, "data", "preprocessed", "{}_cqcc_{}_{}_{}_{}_{}"
                            .format(data_type, B, d, cf, ZsdD, balanced)))

    dataloader = torch.utils.data.DataLoader(cqcc_data, batch_size=batch_size,
                                             num_workers=0, shuffle=is_train, drop_last=True)

    return dataloader


def load_single_metadata(batch_size=50, length=3 * 16000, num_data=10000, data_type="train", custom_path="./data",
                         multi_proc=True):
    option = OPTION

    shuffle = data_type == "train"

    local_dir = os.path.dirname(__file__)

    if os.path.exists(os.path.join(local_dir,
                                   "data/preprocessed/{}_{}_{}_metadata.torch".format(data_type, option, num_data))):
        data = torch.load(os.path.join(local_dir,
                                       "data/preprocessed/{}_{}_{}_metadata.torch".format(data_type, option, num_data)))
        dataloader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=shuffle, drop_last=True)
        return dataloader

    if not os.path.isdir(os.path.join(local_dir, 'data/preprocessed')):
        os.makedirs(os.path.join(local_dir, 'data/preprocessed'))

    if data_type == "train":
        data = ASVDataset(length=length, nb_samples=num_data, random_samples=True, metadata=True,
                          custom_path=custom_path)
    else:
        data = ASVDataset(length=length, is_train=False, is_eval=False, nb_samples=num_data, random_samples=True,
                          metadata=True, custom_path=custom_path)

    print("preprocessing_tools {} set".format(data_type))
    pp_data = PreprocessedASVDataset(data, multi_proc=multi_proc, balanced=False)
    torch.save(pp_data, os.path.join(local_dir,
                                     "data/preprocessed/{}_{}_{}_metadata.torch".format(data_type, option, num_data)))
    dataloader = DataLoader(pp_data, batch_size=batch_size, num_workers=4, shuffle=shuffle, drop_last=True)

    return dataloader
