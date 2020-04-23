import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D


class GradientSolver(nn.Module):
    def __init__(self, specs):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(1e-2), requires_grad=True)
        self.specs = specs
        for o, order in enumerate(specs):
            for i in range(order):
                name = "param_" + str(o) + "_" + str(i)
                setattr(self, name, torch.nn.Parameter(torch.tensor(0.), requires_grad=True))

    def forward(self, x, debug=False):
        assert x.shape[1] == len(self.specs)

        y = self.a
        for (feature, order) in enumerate(self.specs):
            factor = 1.
            for i in range(order):
                name = "param_" + str(feature) + "_" + str(i)
                factor += getattr(self, name) * x[:, feature] ** (i+1)
            y = factor * y
        if debug:
            print("a=", self.a.detach().item())
            print("b=", self.param_0_0.detach().item())
            print("c=", self.param_1_0.detach().item())
            print("x=", x)
            print("expected y = ", self.a.detach().item() * (1 + x[0, 0].detach().item() * self.param_0_0.detach().item()) * (1 + x[0, 1].detach().item() * self.param_1_0.detach().item()))
            print("got y = ", y.detach().item())
        return y


def optimize_params(df, device="vanilla", multiproc=False):
    df = df.loc[df["P"] == 100]
    df = df.loc[df["D"] == 0.5]
    df = df.loc[df["device"] == device]

    X = torch.tensor(np.array([df["I"], df["N"]]).T, dtype=torch.float32)
    specs = [1, 1]

    if multiproc:
        y = torch.tensor(np.array(df["M"]), dtype=torch.float32)
    else:
        y = torch.tensor(np.array(df["S"]), dtype=torch.float32)

    model = GradientSolver(specs)

    lrs = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lrs)
    loss = nn.L1Loss()

    epochs = 30000
    ls = None
    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        ls = loss(model(X), y)

        ls.backward()
        if epoch % 1000 == 0:
            print("loss=", ls.detach().item())
        optimizer.step()

    print(device, multiproc, ls.detach().item(), model.a.detach().item())
    with torch.no_grad():
        model.forward(torch.tensor([[500., 50.]]), debug=True)
    print()
    for parameter in model.parameters():
        print(parameter, parameter)
        print()
    print()
    return model


def evaluate(xy, models):
    Z = np.zeros(xy.shape[0], dtype=np.int32)
    for i, x in tqdm(enumerate(xy), total=len(Z)):
        res = []
        with torch.no_grad():
            for m in models:
                y = m.forward(torch.tensor(x, dtype=torch.float32).view(1, 2))
                res.append(y.item())
        if np.sum(np.abs(x - np.array([500., 50]))) < 20:
            print(x, res)
        Z[i] = np.argmin(np.array(res))
    return Z


def evaluate_single(x, y, model):
    Z = np.zeros(len(x) * len(y), dtype=np.float32)
    for i, x in enumerate(product(x, y)):
        with torch.no_grad():
            y = model.forward(torch.tensor(x, dtype=torch.float32).view(1, 2))
        Z[i] = y.item()
    return Z


def plot_decision(df):
    devices = ["vanilla", "cpu", "cuda"]
    procs = ["S", "M"]

    def get_name(t):
        return t[0] + " " + t[1]

    model_config = list(product(devices, procs))
    model_config.pop(-1)
    models = []

    for config in model_config:
        device, proc = config
        multiproc = True if proc == "M" else False
        models.append(optimize_params(df, device=device, multiproc=multiproc))

    colors = ['#FF7777', '#77FF77', '#7777FF', '#77CCCC', '#CC77CC']
    cmap_light = ListedColormap(colors)
    I_min, I_max = 1., 10 ** 4
    N_min, N_max = 1., 10 ** 2
    n = 50

    ri = (I_max / I_min) ** (1. / n)
    xx = np.arange(n)
    xx = I_min * ri ** xx

    rn = (N_max / N_min) ** (1. / n)
    yy = np.arange(n)
    yy = N_min * rn ** yy

    xx, yy = np.meshgrid(xx, yy)

    Z = evaluate(np.c_[xx.ravel(), yy.ravel()], models)
    Z[0] = 0
    Z[1] = 1
    Z[2] = 2
    Z[3] = 3
    Z[4] = 4
    print(Z)
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    proxy = [plt.Rectangle((0, 0), 1, 1, fc=color)
             for color in colors]

    plt.legend(proxy, [get_name(e) for e in model_config])

    plt.xlabel("I")
    plt.ylabel("N")
    plt.title("best timing for a couple (I, N)")
    plt.show()


def load_data():
    return pd.read_csv("time_stats_local.csv")


def plot_3D(df, device=("vanilla",), fixed_values=None, proc=("M", "S"), show_model=False):
    if fixed_values is None:
        fixed_values = {"P": 100, "D": 0.5}
    fig = plt.figure()
    ax = Axes3D(fig)
    for d in device:
        df_ = df.loc[df["device"] == d]
        df_ = df_.loc[df["P"] == fixed_values["P"]]
        df_ = df_.loc[df["D"] == fixed_values["D"]]

        X = np.array(df_["I"].unique())
        Y = np.array(df_["N"].unique())
        Zs = []
        for p in proc:
            Zs.append(df_[p].to_numpy().reshape(len(Y), len(X)))
        XX, YY = np.meshgrid(X, Y)

        for i, p in enumerate(proc):
            surf = ax.plot_surface(XX, YY, Zs[i], label=p+" "+d, alpha=0.6)
            surf._facecolors2d = surf._facecolors3d
            surf._edgecolors2d = surf._edgecolors3d

        if show_model:
            X = np.linspace(X.min(), X.max(), 100)
            Y = np.linspace(Y.min(), Y.max(), 100)
            XX, YY = np.meshgrid(X, Y)
            for p in proc:
                m = True if p == "M" else "S"
                model = optimize_params(df_, device=d, multiproc=m)
                Z_ = evaluate_single(X, Y, model).reshape(len(Y), len(X))
                surf = ax.plot_surface(XX, YY, Z_, label=p+" "+d+" model", alpha=0.6)
                surf._facecolors2d = surf._facecolors3d
                surf._edgecolors2d = surf._edgecolors3d

    ax.set_xlabel('I')
    ax.set_ylabel('N')
    ax.set_zlabel('time')
    ax.legend()
    plt.show()


def plot_marginal(df, feature, other_features, device="vanilla"):
    # feature = "N"

    for f in other_features:
        if f != feature:
            df = df.loc[df[f] == other_features[f]]
    df = df.loc[df["device"] == device]

    print(df.head())

    X = np.array(df[feature]).reshape(-1, 1)

    ym = df["M"]
    ys = df["S"]

    if device != "cuda":
        plt.plot(X, ym, label='M')
    plt.plot(X, ys, label='S')
    plt.xlabel(feature)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_frame = load_data()

    # N, D, I, P
    # plot_marginal(data_frame, "I", {"P": 100, "D": 0.5, "N": 100, "I": 1000}, device="vanilla")
    # plot_marginal(data_frame, "I", {"P": 100, "D": 0.5, "N": 100, "I": 1000}, device="cpu")

    plot_3D(data_frame, device=("cpu", "vanilla", "cuda"), fixed_values={"P": 100, "D": 0.5}, proc=("S",), show_model=False)
    plot_3D(data_frame, device=("vanilla", ), fixed_values={"P": 100, "D": 0.5}, proc=("M", "S"), show_model=False)
    plot_3D(data_frame, device=("cpu", ), fixed_values={"P": 100, "D": 0.5}, proc=("M", "S"), show_model=False)
    plot_3D(data_frame, device=("cpu", "vanilla"), fixed_values={"P": 100, "D": 0.5}, proc=("M",), show_model=False)

    # plot_marginal(data_frame, "I", {"P": 100, "D": 0.5, "N": 100, "I": 1000}, device="vanilla")
    # plot_marginal(data_frame, "I", {"P": 100, "D": 0.5, "N": 100, "I": 1000}, device="cpu")
    # plot_decision(data_frame)
