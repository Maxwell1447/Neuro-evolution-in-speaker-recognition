import os
import neat
import numpy as np
import random as rd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import webrtcvad
import soundfile as sf
import contextlib
import wave

from anti_spoofing.utils_ASV import SAMPLING_RATE
from anti_spoofing.data_utils import ASVDataset


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def detect_speech_aggressiveness(pcm_audio, nb_frames, nb_elements, aggressiveness: int, sample_rate=16000):
    vad = webrtcvad.Vad(aggressiveness)
    silence_agg = np.zeros(nb_frames)
    for nb_frame in range(nb_frames):
        silence_agg[nb_frame] = vad.is_speech(pcm_audio[nb_frame * nb_elements:(nb_frame + 1) * nb_elements],
                                              sample_rate)
    return silence_agg


def detect_speech(audio, name, frame_duration: int = 10):
    sf.write(name, data=audio, samplerate=SAMPLING_RATE, subtype="PCM_16")
    pcm_audio, sample_rate = read_wave(name)
    nb_elements = 2 * int(sample_rate * frame_duration / 1000)
    nb_frames = int(audio.size // nb_elements)

    silence_0 = detect_speech_aggressiveness(pcm_audio, nb_frames, nb_elements, 0)
    silence_1 = detect_speech_aggressiveness(pcm_audio, nb_frames, nb_elements, 1)
    silence_2 = detect_speech_aggressiveness(pcm_audio, nb_frames, nb_elements, 2)
    silence_3 = detect_speech_aggressiveness(pcm_audio, nb_frames, nb_elements, 3)

    silence = 1 / 4 * (silence_0 + silence_1 + silence_2 + silence_3)

    return silence, nb_frames, nb_elements


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)

    nb_samples = 1

    dev_border = [0, 2548, 6264, 9980, 13696, 17412, 21128, 22296]
    index_test = []
    for i in range(len(dev_border) - 1):
        index_test += rd.sample([k for k in range(dev_border[i], dev_border[i + 1])], nb_samples)

    test_loader = ASVDataset(None, is_train=False, is_eval=False, index_list=index_test)

    silence_prob, nb_frames, nb_elements = detect_speech(test_loader.__getitem__(0)[0], "bonafide.wav")

    t = np.linspace(0, nb_frames * nb_elements, nb_frames)

    sns.set(style="darkgrid")
    plt.plot(t, silence_prob, 'g')
    plt.show()
