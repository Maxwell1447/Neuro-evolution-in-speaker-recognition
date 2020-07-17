from anti_spoofing.data_utils import ASVDataset
from anti_spoofing.constants import *
import torch
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
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

        with Pool(multiprocessing.cpu_count()) as pool:
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


def load_data(batch_size=50, length=3 * 16000):
    """
    loads the data and puts it in PyTorch DataLoader.
    Librispeech uses Index caching to access the data more rapidly.

    If a data loader has not been saved already,
    a data loader is created, then saved for train and test sets.
    """
    option = OPTION

    if os.path.exists("./data/preprocessed/train_{}_{}".format(option, batch_size)) and \
            os.path.exists("./data/preprocessed/test_{}_{}".format(option, batch_size)):
        train_loader = torch.load("./data/preprocessed/train_{}_{}".format(option, batch_size))
        test_loader = torch.load("./data/preprocessed/test_{}_{}".format(option, batch_size))
        return train_loader, test_loader

    if not os.path.isdir('./data/preprocessed'):
        local_dir = os.path.dirname(__file__)
        os.makedirs(os.path.join(local_dir, 'data/preprocessed'))

    trainset = ASVDataset(length=length)
    testset = ASVDataset(length=length, is_train=False, is_eval=True)

    trainset = PreprocessedASVDataset(trainset)
    testset = PreprocessedASVDataset(testset)

    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=1, num_workers=4, drop_last=True, shuffle=False)

    torch.save(train_loader, "./data/preprocessed/train_{}_{}".format(option, batch_size))
    torch.save(test_loader, "./data/preprocessed/test_{}_{}".format(option, batch_size))

    return train_loader, test_loader



