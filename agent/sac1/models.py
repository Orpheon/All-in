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


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

  def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
    super().__init__()
    self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
    self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
    self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
    self.act_limit = act_limit

  def forward(self, obs, deterministic=False, with_logprob=True):
    net_out = self.net(obs)
    mu = self.mu_layer(net_out)
    log_std = self.log_std_layer(net_out)
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = torch.exp(log_std)

    # Pre-squash distribution and sample
    pi_distribution = Normal(mu, std)
    if deterministic:
      # Only used for evaluating policy at test time.
      pi_action = mu
    else:
      pi_action = pi_distribution.rsample()

    if with_logprob:
      # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
      # NOTE: The correction formula is a little bit magic. To get an understanding
      # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
      # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
      # Try deriving it yourself as a (very difficult) exercise. :)
      logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
      logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
    else:
      logp_pi = None

    pi_action = torch.tanh(pi_action)
    pi_action = self.act_limit * pi_action

    return pi_action, logp_pi


class MLPQFunction(nn.Module):

  def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
    super().__init__()
    self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation, output_activation=nn.Tanh)

  def forward(self, obs, act):
    q = self.q(torch.cat([obs, act], dim=-1))
    return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

  def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256,256,256,256),
         activation=nn.LeakyReLU, trainable=True, device="cpu"):
    super().__init__()

    self.trainable = trainable
    self.device = device

    # build policy and value functions
    self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
    self.pi.to(device)
    if self.trainable:
      self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
      self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
      self.q1.to(device)
      self.q2.to(device)
    else:
      self.pi.eval()

  def act(self, obs, deterministic=False):
    with torch.no_grad():
      a, _, = self.pi(obs.to(self.device), deterministic, False)
      return a.cpu().numpy()

  def save(self, path, policy_only=False):
    os.makedirs(path, exist_ok=True)
    torch.save(self.pi.state_dict(), os.path.join(path, 'policy.modelb'))
    if not policy_only:
      torch.save(self.q1.state_dict(), os.path.join(path, 'q1.modelb'))
      torch.save(self.q2.state_dict(), os.path.join(path, 'q2.modelb'))

  def load(self, path):
    self.pi.load_state_dict(torch.load(os.path.join(path, 'policy.modelb')))
    if self.trainable:
      self.q1.load_state_dict(torch.load(os.path.join(path, 'q1.modelb')))
      self.q2.load_state_dict(torch.load(os.path.join(path, 'q2.modelb')))
      self.pi.train()
      self.q1.train()
      self.q2.train()
    else:
      self.pi.eval()