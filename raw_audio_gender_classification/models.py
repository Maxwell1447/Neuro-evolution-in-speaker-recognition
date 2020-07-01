import torch.nn as nn
import torch.nn.functional as F
import torch


class DilatedNet(nn.Module):
    def __init__(self, filters, dilation_depth, dilation_stacks):
        super(DilatedNet, self).__init__()
        self.filters = filters
        self.dilation_depth = dilation_depth
        self.dilation_stacks = dilation_stacks
        self.receptive_field = dilation_stacks*(3**dilation_depth)

        self.initialconv = nn.Conv1d(1, filters, 3, dilation=1, padding=1)

        for s in range(dilation_stacks):
            for i in range(dilation_depth):
                setattr(
                    self,
                    'dilated_conv_{}_relu_s{}'.format(3 ** i, s),
                    nn.Conv1d(filters, filters, 3, dilation=3 ** i, padding=3 ** i)
                )

        self.finalconv = nn.Conv1d(filters, filters, 3, dilation=1, padding=1)
        self.output = nn.Linear(filters, 1)

    def forward(self, x):
        x = x.cuda().double()
        x = self.initialconv(x)

        skip_connections = []
        for s in range(self.dilation_stacks):
            for i in range(self.dilation_depth):
                original_x = x
                x = F.relu(getattr(self, 'dilated_conv_{}_relu_s{}'.format(3 ** i, s))(x))
                skip_connections.append(x)
                x = x + original_x

        x = F.relu(self.finalconv(x))

        x = F.max_pool1d(x, kernel_size=x.size()[2:])
        x = x.view(-1, self.filters)
        x = torch.sigmoid(self.output(x))
        return x


class ConvNet(nn.Module):
    def __init__(self, filters, layers):
        super(ConvNet, self).__init__()
        self.filters = filters
        self.layers = layers
        self.receptive_field = 3**layers

        self.initialconv = nn.Conv1d(1, filters, 3, dilation=1, padding=1)
        self.initialbn = nn.BatchNorm1d(filters)

        for i in range(layers):
            setattr(
                self,
                'conv_{}'.format(i),
                nn.Conv1d(filters, filters, 3, dilation=1, padding=1)
            )
            setattr(
                self,
                'bn_{}'.format(i),
                nn.BatchNorm1d(filters)
            )

        self.finalconv = nn.Conv1d(filters, filters, 3, dilation=1, padding=1)

        self.output = nn.Linear(filters, 1)

    def forward(self, x):
        x = x.cuda().double()
        x = self.initialconv(x)
        x = self.initialbn(x)

        for i in range(self.layers):
            x = F.relu(getattr(self, 'conv_{}'.format(i))(x))
            x = getattr(self, 'bn_{}'.format(i))(x)
            x = F.max_pool1d(x, kernel_size=2, stride=2)

        x = F.relu(self.finalconv(x))

        x = F.max_pool1d(x, kernel_size=x.size()[2:])
        x = x.view(-1, self.filters)
        x = torch.sigmoid(self.output(x))

        return x


class BasicRNN(nn.Module):
    def __init__(self, hidden_size, batch_size, device="cuda"):
        super(BasicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.hx = torch.randn(1, batch_size, hidden_size, device=device)  # initialize hidden state
        self.fc = nn.Linear(self.hidden_size, 2)

    def forward(self, x: torch.Tensor):
        x = x.float()

        xt = x.transpose(0, 1).view(-1, self.batch_size, 1) # seq_len x batch_size x 1

        # Passing in the input and hidden state into the model and obtaining outputs
        out, h_n = self.rnn(xt) # out: seq_len x batch_size x hidden_size

        out = torch.sigmoid(self.fc(out)).transpose(0, 1)  # batch_size x seq_len x 2

        score = out[:, :, 1]
        selectivity = out[:, :, 0]

        selected_score = score * selectivity
        prediction = selected_score.sum(dim=1) / selectivity.sum(dim=1)

        return prediction


class RNN(BasicRNN):
    def __init__(self, n_inputs, hidden_size, batch_size, device="cuda"):
        super(RNN, self).__init__(hidden_size, batch_size, device="cuda")
        self.rnn = nn.RNN(input_size=n_inputs, hidden_size=self.hidden_size, batch_first=True)


class LSTM(BasicRNN):
    def __init__(self, n_inputs, hidden_size, batch_size, device="cuda"):
        super(LSTM, self).__init__(hidden_size, batch_size, device="cuda")
        self.rnn = nn.LSTM(input_size=n_inputs, hidden_size=self.hidden_size, batch_first=True)


class GRU(BasicRNN):
    def __init__(self, n_inputs, hidden_size, batch_size, device="cuda"):
        super(GRU, self).__init__(hidden_size, batch_size, device="cuda")
        self.rnn = nn.GRU(input_size=n_inputs, hidden_size=self.hidden_size, batch_first=True)
