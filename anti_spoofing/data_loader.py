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
        self.meta = torch.empty(self.len, dtype=torch.int64)

        with Pool(multiprocessing.cpu_count()-1) as pool:
            jobs = []

            for i in tqdm(range(self.len)):
                x, t, meta = dataset[i]
                jobs.append(pool.apply_async(preprocess_function, [x.astype(np.float32)]))
                self.t[i] = float(t)
                self.meta[i] = int(meta)

            for i, job in enumerate(jobs):
                self.X[i] = job.get(timeout=None)

    def __getitem__(self, index):
        return self.X[index], self.t[index], self.meta[index]

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



