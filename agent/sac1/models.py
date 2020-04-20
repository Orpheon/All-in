
import torch

class Policy(torch.nn.Module):
  def __init__(self, obs_dim, act_dim):
    super(Policy).__init__()

    self.net = torch.nn.Sequential(
      torch.nn.Linear(obs_dim, obs_dim),
      torch.nn.Linear(obs_dim, obs_dim),
      torch.nn.Linear(obs_dim, act_dim),
    )

  def forward(self, x):
    return self.net(x)


class Qfn(torch.nn.Module):
  def __init__(self, obs_dim):
    super(Qfn).__init__()

    self.net = torch.nn.Sequential(
      torch.nn.Linear(obs_dim, obs_dim),
      torch.nn.Linear(obs_dim, obs_dim),
      torch.nn.Linear(obs_dim, 1),
    )

  def forward(self, x):
    return self.net(x)