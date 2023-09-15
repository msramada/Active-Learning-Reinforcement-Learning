import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable


torch.set_default_dtype(torch.float64)
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.OneFunc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, output_size)
        )
    

    def forward(self, state, action):
        self.eval()
        x = torch.cat((state, action),dim=1)
        x = self.OneFunc(x)
        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.OneFunc = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
 
    def forward(self, state):
        self.eval()
        state = state.to(torch.float64)
        return 5 * self.OneFunc(state)

