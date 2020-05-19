import numpy as np
import torch

# Source: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py

class ReplayBuffer:
  """
  A simple FIFO experience replay buffer for SAC agents.
  """

  def __init__(self, obs_dim, act_dim, size, batch_size, device="cpu"):
    self.obs_buf = torch.zeros(size=(size, batch_size, obs_dim), dtype=torch.float32, requires_grad=False).to(device)
    self.obs2_buf = torch.zeros(size=(size, batch_size, obs_dim), dtype=torch.float32, requires_grad=False).to(device)
    self.act_buf = torch.zeros(size=(size, batch_size, act_dim), dtype=torch.float32, requires_grad=False).to(device)
    self.active_buf = torch.zeros(size=(size, batch_size), dtype=torch.float32, requires_grad=False).to(device)
    self.ptr, self.size, self.max_size = 0, 0, size
    self.replay_ptr = 0
    self.batch_size = batch_size

  def store(self, obs, act, next_obs, active_games):
    if self.ptr > self.max_size:
      self.ptr = 0
      print("WARNING: Replay buffer overflowed")
    self.obs_buf[self.ptr] = torch.Tensor(obs)
    self.obs2_buf[self.ptr] = torch.Tensor(next_obs)
    self.act_buf[self.ptr] = torch.Tensor(act)
    self.active_buf[self.ptr] = torch.Tensor(active_games)

    self.ptr += 1
    self.size = min(self.size+1, self.max_size)

  def sample_state(self):
    if self.replay_ptr <= self.size:
      batch = dict(
        obs=self.obs_buf[self.replay_ptr],
        obs2=self.obs2_buf[self.replay_ptr],
        act=self.act_buf[self.replay_ptr],
        active=self.active_buf[self.replay_ptr])
      self.replay_ptr += 1
      return batch
    else:
      return None