import numpy as np
import torch

# Source: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py

class ReplayBuffer:
  """
  A simple FIFO experience replay buffer for SAC agents.
  """

  def __init__(self, obs_dim, act_dim, size, device="cpu"):
    self.obs_buf = torch.zeros(size=(size, obs_dim), dtype=torch.float32, requires_grad=False).to(device)
    self.obs2_buf = torch.zeros(size=(size, obs_dim), dtype=torch.float32, requires_grad=False).to(device)
    self.act_buf = torch.zeros(size=(size, act_dim), dtype=torch.float32, requires_grad=False).to(device)
    self.rew_buf = torch.zeros(size=(size,), dtype=torch.float32, requires_grad=False).to(device)
    self.done_buf = torch.zeros(size=(size,), dtype=torch.float32, requires_grad=False).to(device)
    self.ptr, self.size, self.max_size = 0, 0, size
    self.replay_ptr = 0

  def store(self, obs, act, rew, next_obs, done, batch_size):
    if self.ptr + batch_size > self.max_size:
      self.shuffle()
      self.ptr = 0
      print("WARNING: Replay buffer overflowed")
    self.obs_buf[self.ptr:self.ptr+batch_size] = torch.Tensor(obs)
    self.obs2_buf[self.ptr:self.ptr+batch_size] = torch.Tensor(next_obs)
    self.act_buf[self.ptr:self.ptr+batch_size] = torch.Tensor(act)
    self.rew_buf[self.ptr:self.ptr+batch_size] = torch.Tensor(rew)
    self.done_buf[self.ptr:self.ptr+batch_size] = torch.Tensor(done)

    self.ptr = (self.ptr+batch_size) % self.max_size
    self.size = min(self.size+batch_size, self.max_size)

  def shuffle(self):
    rand_indices = np.random.permutation(self.obs_buf.shape[0])
    self.obs_buf = self.obs_buf[rand_indices]
    self.obs2_buf = self.obs2_buf[rand_indices]
    self.act_buf = self.act_buf[rand_indices]
    self.rew_buf = self.rew_buf[rand_indices]
    self.done_buf = self.done_buf[rand_indices]
    self.replay_ptr = 0

  def sample_batch(self, batch_size):
    if self.replay_ptr + batch_size <= self.size:
      batch = dict(obs=self.obs_buf[self.replay_ptr:self.replay_ptr+batch_size],
             obs2=self.obs2_buf[self.replay_ptr:self.replay_ptr+batch_size],
             act=self.act_buf[self.replay_ptr:self.replay_ptr+batch_size],
             rew=self.rew_buf[self.replay_ptr:self.replay_ptr+batch_size],
             done=self.done_buf[self.replay_ptr:self.replay_ptr+batch_size])
      self.replay_ptr += batch_size
      return batch
    else:
      return None