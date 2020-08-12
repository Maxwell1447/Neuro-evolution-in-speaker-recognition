import numpy as np
import librosa
from spafe.features.lfcc import lfcc
import torch


def preprocess(y, option="stft", bins=24, sr=16000, win_length=1024, hop_length=512):
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
        z = lfcc(sig=y, fs=sr, num_ceps=20, pre_emph=0, pre_emph_coeff=0.97, win_len=0.030, win_hop=0.015,
                 win_type="hamming", nfilts=70, nfft=1024, low_freq=0, high_freq=8000, scale="constant",
                 dct_type=2, use_energy=False, lifter=22, normalize=0)
    else:
        raise ValueError("option ill-defined")

    return z
