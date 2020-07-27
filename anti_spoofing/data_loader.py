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

        with Pool(multiprocessing.cpu_count()-1) as pool:
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
    option = OPTION

    if os.path.exists("./data/preprocessed/train_{}_{}_{}".format(option, batch_size, num_train)) and \
            os.path.exists("./data/preprocessed/test_{}_{}_{}".format(option, batch_size, num_test)):
        train_loader = torch.load("./data/preprocessed/train_{}_{}_{}".format(option, batch_size, num_train))
        test_loader = torch.load("./data/preprocessed/test_{}_{}_{}".format(option, batch_size, num_test))
        return train_loader, test_loader

    if not os.path.isdir('./data/preprocessed'):
        local_dir = os.path.dirname(__file__)
        os.makedirs(os.path.join(local_dir, 'data/preprocessed'))

    trainset = ASVDataset(length=length, nb_samples=num_train, random_samples=True)
    testset = ASVDataset(length=length, is_train=False, is_eval=False, nb_samples=num_test, random_samples=True)

    print("preprocessing train set")
    trainset = PreprocessedASVDataset(trainset)
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    torch.save(train_loader, "./data/preprocessed/train_{}_{}_{}".format(option, batch_size, num_train))
    del trainset

    print("preprocessing test set")
    testset = PreprocessedASVDataset(testset)
    test_loader = DataLoader(testset, batch_size=1, num_workers=4, drop_last=True, shuffle=False)
    torch.save(test_loader, "./data/preprocessed/test_{}_{}_{}".format(option, batch_size, num_test))
    del testset

    return train_loader, test_loader


def load_data_cqcc(batch_size=50, num_train=1000, num_test=1000):
    """
    loads the data and puts it in PyTorch DataLoader.
    Librispeech uses Index caching to access the data more rapidly.

    If a data loader has not been saved already,
    a data loader is created, then saved for train and test sets.
    """

    local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "anti_spoofing")

    if os.path.exists(os.path.join(local_dir, "data", "preprocessed", "train_cqcc_{}_{}_{}_{}".format(B, d, cf, ZsdD))) and \
            os.path.exists(os.path.join(local_dir, "data", "preprocessed", "dev_cqcc_{}_{}_{}_{}".format(B, d, cf, ZsdD))):

        print("data found locally")
        train_loader = torch.load("./data/preprocessed/train_cqcc_{}_{}_{}_{}".format(B, d, cf, ZsdD))
        test_loader = torch.load("./data/preprocessed/dev_cqcc_{}_{}_{}_{}".format(B, d, cf, ZsdD))
        return train_loader, test_loader

    if not os.path.isdir('./data/preprocessed'):
        local_dir = os.path.dirname(__file__)
        os.makedirs(os.path.join(local_dir, 'data/preprocessed'))

    print("loading .mat files to PyTorch...")

    cqcc_train = CQCCDataset(params_id="train_2048_2048_19_Zs", n_files=num_train)
    train_dataloader = torch.utils.data.DataLoader(cqcc_train, batch_size=batch_size, num_workers=multiprocessing.cpu_count())
    torch.save(train_dataloader, os.path.join(local_dir, "data", "preprocessed", "train_cqcc_{}_{}_{}_{}".format(B, d, cf, ZsdD)))

    cqcc_dev = CQCCDataset(params_id="dev_2048_2048_19_Zs", n_files=num_test)
    dev_dataloader = torch.utils.data.DataLoader(cqcc_dev, batch_size=1, num_workers=multiprocessing.cpu_count())
    torch.save(dev_dataloader, os.path.join(local_dir, "data", "preprocessed", "dev_cqcc_{}_{}_{}_{}".format(B, d, cf, ZsdD)))

    return train_dataloader, dev_dataloader



