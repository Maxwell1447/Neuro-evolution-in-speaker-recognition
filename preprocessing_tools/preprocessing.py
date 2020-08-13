import numpy as np
import librosa
from spafe.features.lfcc import lfcc
from anti_spoofing.constants import *
import torch


def preprocess(y, option=OPTION, bins=BINS, sr=16000, win_length=WIN_LEN, hop_length=HOP_LEN):
    # y seq_len

    if option == "stft":
        z = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=bins, n_fft=win_length,
                                        hop_length=hop_length).T  # seq_len/hop_len x bins
    elif option == "cqt":
        z = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=bins, hop_length=hop_length).T  # seq_len/hop_len x bins
    elif option == "mfcc":
        z = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=bins, n_fft=win_length,
                                 hop_length=hop_length).T  # seq_len/hop_len x bins
    elif option == "lfcc":
        z = lfcc(sig=y, fs=sr, num_ceps=bins, pre_emph=0, pre_emph_coeff=0.97, win_len=win_length/16000,
                 win_hop=hop_length/16000,
                 win_type="hamming", nfilts=70, nfft=win_length, low_freq=0, high_freq=8000, scale="constant",
                 dct_type=2, use_energy=False, lifter=22, normalize=0)
    else:
        raise ValueError("option {} not defined".format(option))

    return z
