from raw_audio_gender_classification.models import *
from simple_audio.audio_generator import signal
import random as rd
import torch
import torch.optim as optim
from raw_audio_gender_classification.utils import whiten
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

BATCH_SIZE = 20


def get_batch(batch_size=BATCH_SIZE):

    n = 50
    
    X = torch.empty(batch_size, n)
    y = torch.empty(batch_size, dtype=torch.bool)
    for b in range(batch_size):
        A = 1.
        f = 10 ** (rd.random() * 2 - 3)
        y[b] = f > 10**-2
        X[b] = torch.from_numpy(signal(n=n, f=f, magnitude=A))

    return X, y


if __name__ == "__main__":

    model_name = "LSTM"

    if model_name == "LSTM":
        rnn = LSTM(1, 300, BATCH_SIZE, device="cpu")
    elif model_name == "RNN":
        rnn = RNN(1, 300, BATCH_SIZE, device="cpu")
    elif model_name == "GRU":
        rnn = GRU(1, 300, BATCH_SIZE, device="cpu")
    elif model_name == "ConvNet":
        rnn = ConvNet(64, 4)
        rnn.double().cuda()
    else:
        raise ValueError("wrong model used")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(rnn.parameters(), lr=0.001)

    writer = SummaryWriter('./runs/experiment_rnn')
    print(writer.log_dir)


    def get_accuracy(logit, target):
        """ Obtain accuracy for training round """
        corrects = (torch.abs(logit - target) < 0.5).sum()
        accuracy = 100.0 * corrects / BATCH_SIZE
        return accuracy.item()


    for batch in tqdm(range(1000)):

        rnn.train()

        # zero the parameter gradients
        optimizer.zero_grad()

        # get the inputs
        inputs, labels = get_batch()

        labels = labels.view(BATCH_SIZE, 1)  # .cuda()
        inputs = whiten(inputs)

        if model_name == "ConvNet":
            inputs = inputs.view(BATCH_SIZE, 1, -1).double()
            labels = labels.cuda().double()
        else:
            labels = labels.float()

        # forward + backward + optimize
        outputs = rnn(inputs).view(BATCH_SIZE, 1)
        jitter = 1e-10
        outputs = outputs.clamp(jitter, 1.-jitter)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        current_loss = loss.detach().item()
        writer.add_scalar('training loss - {} 50'.format(model_name), current_loss)
        train_acc = get_accuracy(outputs, labels)

        rnn.eval()
        print('Batch:  %d | Loss: %.4f | Train Accuracy: %.2f'
              % (batch, current_loss, train_acc))
