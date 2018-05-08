import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

KEY_STR = '0x90e12f6e390b52a09eec5e8bbf02502b'


class Blockhash(nn.Module):

    def __init__(self, iterations=50, key=KEY_STR):
        self.iterations = iterations
        key_data = [self._quantized(int(key[i:i + 8], 16))
                    for i in range(2, 27, 8)]
        self.key_variable = Variable(torch.FloatTensor(key_data))

    def _step(self, input_):
        return self._out(self._features(input_))

    def _quantized(self, num):
        return num / 4294967296

    def _chaotic_map(self, x1, x2, T=1):
        for _ in range(T):
            if 0 <= x1 < x2:
                x1 = x1 / x2
            elif x2 <= x1 < 0.5:
                x1 = (x1 - x2) / (0.5 - x2)
            elif 0.5 <= x1 < 1 - x2:
                x1 = (1 - x2 - x1) / (0.5 - x2)
            elif 1 - x2 <= x1 <= 1:
                x1 = (1 - x1) / x2
        return x1

    def _key_generation(self, key_tensor):
        k0, k1, k2, k3 = key_tensor.data
        subkeys = []
        for i in range(151):
            x0 = self._chaotic_map(k0, k1,self.iterations+i)
            x1 = self._chaotic_map(k2, k3,self.iterations+i)
            ks = x0 + x1
            if 0 <= ks < 1:
                ks = ks
            elif 1 <= ks < 2:
                ks = ks - 1
            subkeys.append(ks)

        w0 = torch.FloatTensor(subkeys[0:32])
        b0 = torch.FloatTensor(subkeys[32:40])
        q0 = torch.FloatTensor([subkeys[40]])
        w1 = torch.FloatTensor([ subkeys[i:i+8] for i in range(41,105,8) ])
        b1 = torch.FloatTensor(subkeys[105:113])
        q1 = torch.FloatTensor([subkeys[113]])
        w2 = torch.FloatTensor([subkeys[i:i+8] for i in range(114,146,8)])
        b2 = torch.FloatTensor(subkeys[146:151])
        q2 = torch.FloatTensor([subkeys[151]])

        return w0,b0,q0,w1,b1,q1,w2,b2,q2

    def _blockhash(self,input,key):
        w0,b0,q0,w1,b1,q1,w2,b2,q2 = self._key_generation(key)
        input = input*w0
        input = input.view(8,4).sum(dim=1)+b0
        c = torch.FloatTensor([self._chaotic_map(i,q0,self.iterations) for i in input])
        c = torch.mv(w1,c)+b1
        d = torch.FloatTensor([self._chaotic_map(i,q1,1) for i in c])
        d = torch.mv(w2,d)+b2
        h = torch.FloatTensor([self._chaotic_map(i,q2,self.iterations) for i in d])
        return h

    def forward(self, input_gen):
        prev_k = self.key_variable
        for chunk in input_gen:
            h = self._blockhash(chunk,prev_k)
            prev_k = [int(i*4294967296) for i in prev_k]
            h = [int(i*4294967296) for i in h]
            prev_k = [self._quantized(prev_k[i]^h[i]) for i in range(4)]
        return prev_k
