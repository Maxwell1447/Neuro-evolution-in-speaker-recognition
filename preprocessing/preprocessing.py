import numpy as np
import librosa
import torch


def preprocess(y, option="stft", bins=24, sr=16000):

    # y seq_len

    if option == "stft":
        z = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=bins, n_fft=1024).T  # seq_len/n_fft/2 x bins
    elif option == "cqt":
        z = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=bins, hop_length=512).T  # seq_len/hop_len x bins
    elif option == "mfcc":
        z = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=bins, hop_length=512).T  # seq_len/hop_len x bins
    else:
        raise ValueError("option ill-defined")

    return z
