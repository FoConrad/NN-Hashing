import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

class RNN(nn.Module):
    def __init__(self, raw_input_size, hidden_sizes, output_size, cudize,
            data_err=False, gen_length=0):
        super(RNN, self).__init__()
        self._raw_input_size = raw_input_size
        self._output_size = output_size
        self._cudize = cudize
        self._data_err = data_err

        act = nn.Tanh()
        last_size = raw_input_size + output_size
        sequentials = []
        for hsize in hidden_sizes:
            sequentials.append(nn.Linear(last_size, hsize))
            sequentials.append(act)
            last_size = hsize
        sequentials.append(nn.Linear(last_size, output_size))

        self._features = nn.Sequential(*sequentials)
        self._out = nn.Sigmoid()
        if data_err:
            self._data_delta = Parameter(
                    data=self._cudize(torch.zeros(gen_length, raw_input_size)))

    def cudize(self):
        return self._cudize(self)

    @property
    def r(self):
        return self._data_delta

    def _step(self, input_):
        return self._out(self._features(input_))

    def forward(self, input_gen, ignore_data_err=True):
        output = Variable(self._cudize(torch.zeros(self._output_size)))
        for index, chunk in enumerate(input_gen):
            input_ = Variable(self._cudize(torch.FloatTensor(chunk)))
            if not ignore_data_err:
                input_ += self._data_delta[index]
            input_ = torch.cat((input_, output))
            output = self._step(input_)
        return output

