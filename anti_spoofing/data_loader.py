from anti_spoofing.data_utils import ASVDataset
from anti_spoofing.constants import *
import torch
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

from anti_spoofing.matlab.mat_loader import CQCCDataset
from preprocessing.preprocessing import preprocess
import os
from torch.utils.data.dataloader import DataLoader


def preprocess_function(s):
    return torch.from_numpy(preprocess(s, option=OPTION, bins=BINS, sr=16000))


class PreprocessedASVDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: ASVDataset):
        self.len = len(dataset)
        self.X = torch.empty(self.len, 16000 * 3 // 512 + 1, BINS, dtype=torch.float32)
        self.t = torch.empty(self.len, dtype=torch.float32)

        with Pool(multiprocessing.cpu_count() - 1) as pool:
            jobs = []

            for i in tqdm(range(self.len)):
                x, t, meta = dataset[i]
                jobs.append(pool.apply_async(preprocess_function, [x.astype(np.float32)]))
                self.t[i] = float(t)

            for i, job in enumerate(jobs):
                self.X[i] = job.get(timeout=None)

    def __getitem__(self, index):
        return self.X[index], self.t[index]

    def __len__(self):
        return self.len


def load_data(batch_size=50, length=3 * 16000, num_train=10000, num_test=10000):
    """
    loads the data and puts it in PyTorch DataLoader.
    Librispeech uses Index caching to access the data more rapidly.

    If a data loader has not been saved already,
    a data loader is created, then saved for train and test sets.
    """

    train_loader = load_single_data(batch_size=batch_size, length=length, num_data=num_train, data_type="train")
    test_loader = load_single_data(batch_size=1, length=length, num_data=num_train, data_type="test")

    return train_loader, test_loader


def load_data_cqcc(batch_size=50, num_train=1000, num_test=1000, balanced=False):
    """
    loads the data and puts it in PyTorch DataLoader.
    Librispeech uses Index caching to access the data more rapidly.

    If a data loader has not been saved already,
    a data loader is created, then saved for train and test sets.
    """

    train_dataloader = load_single_data_cqcc(batch_size=batch_size, num_data=num_train,
                                             balanced=balanced, data_type="train")
    dev_dataloader = load_single_data_cqcc(batch_size=1, num_data=num_test,
                                           balanced=balanced, data_type="test")

    return train_dataloader, dev_dataloader


def load_single_data(batch_size=50, length=3 * 16000, num_data=10000, data_type="train"):
    option = OPTION

    shuffle = data_type == "train"

    if os.path.exists("./data/preprocessed/train_{}_{}".format(option, num_data)):
        data = torch.load("./data/preprocessed/train_{}_{}".format(option, num_data))
        dataloader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=shuffle, drop_last=True)
        return dataloader

    if not os.path.isdir('./data/preprocessed'):
        local_dir = os.path.dirname(__file__)
        os.makedirs(os.path.join(local_dir, 'data/preprocessed'))

    if data_type == "train":
        data = ASVDataset(length=length, nb_samples=num_data, random_samples=True)
    else:
        data = ASVDataset(length=length, is_train=False, is_eval=False, nb_samples=num_data, random_samples=True)

    print("preprocessing {} set".format(data_type))
    pp_data = PreprocessedASVDataset(data)
    torch.save(pp_data, "./data/preprocessed/train_{}_{}".format(option, num_data))
    dataloader = DataLoader(pp_data, batch_size=batch_size, num_workers=4, shuffle=shuffle, drop_last=True)

    return dataloader


def load_single_data_cqcc(batch_size=50, num_data=1000, balanced=False, data_type="train"):
    local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "anti_spoofing")

    shuffle = data_type == "train"

    if os.path.exists(
            os.path.join(local_dir, "data", "preprocessed", "{}_cqcc_{}_{}_{}_{}".format(data_type, B, d, cf, ZsdD))):
        print("{} data found locally".format(data_type))
        cqcc_data = torch.load("./data/preprocessed/{}_cqcc_{}_{}_{}_{}"
                               .format(data_type, B, d, cf, ZsdD))
        dataloader = torch.utils.data.DataLoader(cqcc_data, batch_size=batch_size,
                                                 num_workers=0, shuffle=shuffle)
        return dataloader

    if not os.path.isdir('./data/preprocessed'):
        local_dir = os.path.dirname(__file__)
        os.makedirs(os.path.join(local_dir, 'data/preprocessed'))

    print("loading {} .mat files to PyTorch...".format(data_type))

    cqcc_data_type = data_type if data_type == "train" else "dev"
    cqcc_data = CQCCDataset(params_id="{}_{}_{}_{}_{}".format(cqcc_data_type, B, d, cf, ZsdD),
                            n_files=num_data, balanced=balanced)

    torch.save(cqcc_data,
               os.path.join(local_dir, "data", "preprocessed", "{}_cqcc_{}_{}_{}_{}"
                            .format(data_type, B, d, cf, ZsdD)))

    dataloader = torch.utils.data.DataLoader(cqcc_data, batch_size=batch_size,
                                             num_workers=0, shuffle=shuffle)

    return dataloader
