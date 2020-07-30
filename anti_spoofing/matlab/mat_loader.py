import collections
import scipy.io
import os
import numpy as np
import torch
import multiprocessing
from tqdm import tqdm

local_dir = os.path.dirname(os.path.dirname(__file__))
ASVFile = collections.namedtuple('ASVFile',
                                 ['speaker_id', 'path', 'sys_id', 'key'])

print(local_dir)
PROTOCOLS_PATH = os.path.abspath(os.path.join(local_dir, "data", "LA", "ASVspoof2019_LA_cm_protocols"))


class CQCCDataset(torch.utils.data.Dataset):

    def __init__(self, params_id="train_96_16_19_Zs", n_files=10000, balanced=True):
        directory = os.path.join(local_dir, "data", "features", params_id)

        if not os.path.exists(directory):
            raise NotADirectoryError("trying to fetch data with id "+params_id)

        self.files_dir = directory
        self.fixed_t = 20
        self.n_files = n_files
        self.balanced = balanced

        self.sysid_dict = {
            '-': 0,  # bonafide speech
            'A01': 1,  # Wavenet vocoder
            'A02': 2,  # Conventional vocoder WORLD
            'A03': 3,  # Conventional vocoder MERLIN
            'A04': 4,  # Unit selection system MaryTTS
            'A05': 5,  # Voice conversion using neural networks
            'A06': 6,  # transform function-based voice conversion
            # For PA:
            'A07': 7,
            'A08': 8,
            'A09': 9,
            'A10': 10,
            'A11': 11,
            'A12': 12,
            'A13': 13,
            'A14': 14,
            'A15': 15,
            'A16': 16,
            'A17': 17,
            'A18': 18,
            'A19': 19
        }

        self.data_set_type = params_id.split("_")[0]
        self.meta = {}
        if self.data_set_type == "train":
            ext = 'trn'
        else:
            ext = 'trl'
        self.protocols_fname = os.path.join(PROTOCOLS_PATH, "ASVspoof2019.LA.cm.{}.{}.txt"
                                            .format(self.data_set_type, ext))
        self.parse_protocols_file(self.protocols_fname)

        if balanced:
            matS = []  # spoofed
            matB = []  # bonafide
        else:
            mats = []
        ys = []
        cpt = 0
        for filename in tqdm(os.listdir(directory), total=min(len(os.listdir(directory)), self.n_files)):
            if filename.endswith(".mat"):
                if cpt > self.n_files != -1:
                    break
                f = filename.split('.')[0]
                if balanced:
                    if self.meta[f].key:
                        matB.append(tile_trunc(scipy.io.loadmat(os.path.join(directory, filename))["x"], self.fixed_t))
                    else:
                        matS.append(tile_trunc(scipy.io.loadmat(os.path.join(directory, filename))["x"], self.fixed_t))
                else:
                    mats.append(tile_trunc(scipy.io.loadmat(os.path.join(directory, filename))["x"], self.fixed_t))

                ys.append(self.meta[f].key)
                cpt += 1

        if balanced:
            self.dataS_x = torch.from_numpy(np.array(matS)).float()  # m
            self.dataB_x = torch.from_numpy(np.array(matB)).float()  # n (n<m)
            self.length = 2 * len(self.dataS_x)  # 2m
            print("S", self.dataS_x.shape)
            print("B", self.dataB_x.shape)
        else:
            self.data_x = torch.from_numpy(np.array(mats)).float()
            self.length = len(self.data_x)
            print(self.data_x.shape)

        self.data_y = torch.from_numpy(np.array(ys)).float()

        print(self.data_y.shape)

    def __getitem__(self, item):
        if self.balanced:
            if item < self.length // 2:
                return self.dataS_x[item], 0
                # return self.dataS_x[item], 0
            else:
                n = len(self.dataB_x)
                return self.dataB_x[item % n], 1
                # return self.dataB_x[item % n], 1
        else:
            return self.data_x[item], self.data_y[item]

    def __len__(self):
        return self.length

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        self.meta[tokens[1]] = ASVFile(speaker_id=tokens[0],
                                       path=os.path.join(self.files_dir, tokens[1] + '.flac'),
                                       sys_id=self.sysid_dict[tokens[3]],
                                       key=int(tokens[4] == 'bonafide'))

    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()

        for line in lines:
            self._parse_line(line)


def tile_trunc(x: np.ndarray, t: int):

    r = np.ceil(t / x.shape[1]).astype(np.int)
    
    tiled_x = np.tile(x, r)
    l = tiled_x.shape[1]  # l >= t
    
    offset = np.random.randint(0, l-t+1)

    return tiled_x[:, offset: offset+t]


if __name__ == "__main__":
    cqcc_train = CQCCDataset(params_id="train_2048_2048_19_Zs")
    train_dataloader = torch.utils.data.DataLoader(cqcc_train, batch_size=50, num_workers=multiprocessing.cpu_count())
    torch.save(train_dataloader, os.path.join(local_dir, "data", "preprocessed", "train_cqcc_2048_2048_19_Zs"))

    cqcc_eval = CQCCDataset(params_id="eval_2048_2048_19_Zs")
    eval_dataloader = torch.utils.data.DataLoader(cqcc_eval, batch_size=1, num_workers=multiprocessing.cpu_count())
    torch.save(eval_dataloader, os.path.join(local_dir, "data", "preprocessed", "eval_cqcc_2048_2048_19_Zs"))

    cqcc_dev = CQCCDataset(params_id="dev_2048_2048_19_Zs")
    dev_dataloader = torch.utils.data.DataLoader(cqcc_dev, batch_size=1, num_workers=multiprocessing.cpu_count())
    torch.save(dev_dataloader, os.path.join(local_dir, "data", "preprocessed", "dev_cqcc_2048_2048_19_Zs"))
