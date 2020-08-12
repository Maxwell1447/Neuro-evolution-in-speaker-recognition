import torch
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from anti_spoofing.constants import *
from anti_spoofing.data_loader import load_data_cqcc, load_data
from anti_spoofing.metrics_utils import rocch, rocch2eer


class LinearModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = torch.nn.Linear(20, 2)

    def forward(self, x: torch.Tensor):
        # x: batch_size x t x bins
        xt = x.transpose(0, 1)  # x: t x batch_size x bins

        contribution = torch.zeros(xt.shape[1])
        norm = torch.zeros(xt.shape[1])
        for xi in xt:
            # xi: batch_size x bins

            # Usage of batch evaluation provided by PyTorch-NEAT
            xo = self.single_forward(xi)  # batch_size x 2
            score = xo[:, 1]
            confidence = xo[:, 0]
            contribution += score * confidence  # batch_size
            norm += confidence

        jitter = 1e-8
        prediction = (contribution / (norm + jitter))

        return prediction

    def single_forward(self, x):
        return torch.sigmoid(self.fc(x))


class LinearModel2(LinearModel):

    def __init__(self):
        super().__init__()

        self.fc = torch.nn.Linear(20, 50)
        self.fc2 = torch.nn.Linear(50, 2)

    def single_forward(self, x):
        x = torch.sigmoid(self.fc(x))
        x = torch.sigmoid(self.fc2(x))

        return x


def get_eer_acc(y, out):
    """
    returns the equal error rate and the accuracy
    """

    target_scores = out[y == 1].numpy()
    non_target_scores = out[y == 0].numpy()

    acc = ((y - out).abs() < 0.5).float().mean()

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return eer, acc


def evaluate(model, data):
    scores = []
    targets = []

    for batch in tqdm(iter(data), total=len(data)):
        x, y = batch
        out = model(x)

        scores.append(out.numpy())
        targets.append(y.numpy())

    scores = np.concatenate(scores)
    targets = np.concatenate(targets)

    target_scores = scores[targets == 1]
    non_target_scores = scores[targets == 0]

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    acc = np.mean(np.abs(scores - targets) < 0.5)

    return eer, acc


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'ASV_neat_preprocessed.cfg')

    if OPTION == "cqcc":
        trainloader, devloader = load_data_cqcc(batch_size=100, batch_size_test=100, num_train=10000,
                                                num_test=10000, balanced=True)
        evalloader = None
    else:
        trainloader, devloader, evalloader = load_data(batch_size=100, batch_size_test=100, num_train=10000,
                                                       num_test=10000, balanced=True, include_eval=True)

    writer = SummaryWriter('./runs/{}'.format(OPTION))
    model = LinearModel2()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    for epoch in range(10):
        print("EPOCH ", epoch)
        for batch in tqdm(iter(trainloader), total=len(trainloader)):
            optimizer.zero_grad()
            x, y = batch

            out = model(x)
            loss = criterion(out, y.float())
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                eer, acc = get_eer_acc(y, out)
            writer.add_scalar('training loss', loss.detach().item())
            writer.add_scalar('training accuracy', acc)
            writer.add_scalar('training eer', eer)

    print("TESTING on dev")
    with torch.no_grad():
        eer, acc = evaluate(model, devloader)

    print("DEV EER / ACC:")
    print(eer)
    print(acc)

    if evalloader is not None:
        print("TESTING on eval")
        with torch.no_grad():
            eer, acc = evaluate(model, evalloader)

        print("EVAL EER / ACC:")
        print(eer)
        print(acc)
