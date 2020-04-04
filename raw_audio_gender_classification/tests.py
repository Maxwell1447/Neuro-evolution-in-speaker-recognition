import unittest
import torch
import soundfile as sf
import numpy as np

from raw_audio_gender_classification.utils import whiten
from raw_audio_gender_classification.config import PATH


class TestWhitening(unittest.TestCase):
    def test_whitening(self):
        desired_rms = 0.038021

        test_data, sample_rate = sf.read(PATH + '/data/whitening_test_audio.flac')
        test_data = np.stack([test_data]*2)

        whitened = whiten(torch.from_numpy(test_data), desired_rms)

        # Mean correct
        self.assertTrue(np.isclose(whitened.mean().item(), 0))

        # RMS correct
        self.assertTrue(np.isclose(np.sqrt(np.power(whitened[0,:], 2).mean()).item(), desired_rms))
