import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import os


def mlp(sizes, activation, output_activation=nn.Identity):
  layers = []
  for j in range(len(sizes)-1):
    act = activation if j < len(sizes)-2 else output_activation
    layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
  return nn.Sequential(*layers)

class MLPQFunction(nn.Module):

  def __init__(self,
               obs_dim,
               act_dim,
               hidden_sizes=(256,256,256,256),
               activation=nn.LeakyReLU,
               trainable=True,
               device="cpu"):
    super().__init__()

    self.device = device

    self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation=nn.Tanh)
    self.q.to(device)
    self.trainable = trainable
    if trainable:
      self.q.train()
    else:
      self.q.eval()

  def forward(self, obs):
    return self.q(obs)

  def save(self, path):
    os.makedirs(path, exist_ok=True)
    torch.save(self.q.state_dict(), os.path.join(path, 'q.modelb'))

  def load(self, path):
    self.q.load_state_dict(torch.load(os.path.join(path, 'q.modelb'), map_location=self.device))
    if self.trainable:
      self.q.train()
    else:
      self.q.eval()
    self.q.to(self.device)