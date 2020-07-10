import os
import neat
import numpy as np
import random as rd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import webrtcvad

from tqdm import tqdm

from anti_spoofing.utils import SAMPLING_RATE
from anti_spoofing.data_utils import ASVDataset
from anti_spoofing.silence_detection import detect_speech

path = "neat-checkpoint-39"

nb_samples = 1

dev_border = [0, 2548, 6264, 9980, 13696, 17412, 21128, 22296]
index_test = []
for i in range(len(dev_border) - 1):
    index_test += rd.sample([k for k in range(dev_border[i], dev_border[i + 1])], nb_samples)

vad = webrtcvad.Vad()
vad.set_mode(1)
frame_duration = 10  # ms
frame = b'\x00\x00' * int(SAMPLING_RATE * frame_duration / 1000)
print('Contains speech: %s' % (vad.is_speech(frame, SAMPLING_RATE)))

test_loader = ASVDataset(None, is_train=False, is_eval=False, index_list=index_test)


def whiten(single_input):
    whiten_input = single_input - single_input.mean()
    var = np.sqrt((whiten_input ** 2).mean())
    whiten_input *= 1 / var
    return whiten_input


def gate_activation(recurrent_net, inputs):
    length = inputs.size
    score, select = np.zeros(length), np.zeros(length)
    for i in range(length):
        select[i], score[i] = recurrent_net.activate([inputs[i]])
    mask = (select > 0.5)
    return mask, score


def evaluate(net, data_loader):
    net.reset()
    gates = []
    scores = []
    for data in data_loader:
        inputs, output = data[0], data[1]
        inputs = whiten(inputs)
        mask, score = gate_activation(net, inputs)
        gates.append(mask)
        scores.append(score)

    return np.array(gates), np.array(scores)


def run(config_file, path):
    """
    Launches a run until convergence or max number of generation reached
    :param config_file: path to the config file
    :param n_gen: lax number of generation
    :return: the best genontype (winner), the configs, the stats of the run and the accuracy on the testing set
    """
    # Load configuration.
    config_ = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_file)

    # load saved population
    p = neat.Checkpointer.restore_checkpoint(path)

    genomes = p.population
    nb_genomes = len(genomes)

    gates = []
    scores = []

    for genome_id in tqdm(genomes):
        net = neat.nn.RecurrentNetwork.create(genomes[genome_id], config_)
        gate, score = evaluate(net, test_loader)
        gates.append(gate)
        scores.append(score)

    return np.array(gates), np.array(scores), nb_genomes


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')
    gates, scores, nb_genomes = run(config_path, path)

    audio_samples_using = []
    score_audio_samples_using = []
    speech_detection = []
    speech_detection_time = []
    for audio_sample in range(nb_samples * 7):
        audio_samples_using.append(gates[0][audio_sample].astype('int'))
        score_audio_samples_using.append(scores[0][audio_sample])

        # sum gates dans scores over nb_genomes genomes
        for genome in range(1, nb_genomes):
            audio_samples_using[audio_sample] += gates[genome][audio_sample].astype('int')
            score_audio_samples_using[audio_sample] += scores[genome][audio_sample]

        # to smooth gates
        audio_samples_using[audio_sample] = audio_samples_using[audio_sample] / nb_genomes
        audio_samples_using[audio_sample] = savgol_filter(audio_samples_using[audio_sample], 201, 3)

        # to smooth scores
        score_audio_samples_using[audio_sample] = score_audio_samples_using[audio_sample] / nb_genomes
        score_audio_samples_using[audio_sample] = savgol_filter(score_audio_samples_using[audio_sample], 201, 3)

        # to retrieve detection of speech
        raw_audio_sample = test_loader.__getitem__(audio_sample)
        name = str(raw_audio_sample[2]) + ".wav"
        silence, nb_frames, nb_elements = detect_speech(raw_audio_sample[0], name)
        speech_detection.append(silence)
        speech_detection_time.append((nb_frames, nb_elements))



    sns.set(style="darkgrid")
    for audio_sample in range(nb_samples * 7):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        # plot gates
        ax1.plot(audio_samples_using[audio_sample], 'b', label="average gates")

        # plot scores
        ax2.plot(score_audio_samples_using[audio_sample], 'r', label="average scores")

        # plot detection of speech
        nb_frames, nb_elements = speech_detection_time[audio_sample]
        t = np.linspace(0, nb_frames * nb_elements, nb_frames)
        ax3.plot(t, speech_detection[audio_sample], 'g', label="speech detection")
        fig.legend()
        plt.show()
