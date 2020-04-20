import numpy as np
import torch

# Source: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py

def combined_shape(length, shape=None):
  if shape is None:
    return (length,)
  return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
  """
  A simple FIFO experience replay buffer for SAC agents.
  """

  def __init__(self, obs_dim, act_dim, size):
    self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
    self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
    self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
    self.rew_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.ptr, self.size, self.max_size = 0, 0, size
    self.replay_ptr = 0

  def store(self, obs, act, rew, next_obs, done, batch_size):
    self.obs_buf[self.ptr:self.ptr+batch_size] = obs
    self.obs2_buf[self.ptr:self.ptr+batch_size] = next_obs
    self.act_buf[self.ptr:self.ptr+batch_size] = act
    self.rew_buf[self.ptr:self.ptr+batch_size] = rew
    self.done_buf[self.ptr:self.ptr+batch_size] = done
    self.ptr += batch_size
    if self.ptr + batch_size > self.max_size:
      self.shuffle()
    self.ptr = (self.ptr+batch_size) % self.max_size
    self.size = min(self.size+batch_size, self.max_size)

  def shuffle(self):
    rng_state = np.random.get_state()
    np.random.shuffle(self.obs_buf)
    np.random.set_state(rng_state)
    np.random.shuffle(self.obs2_buf)
    np.random.set_state(rng_state)
    np.random.shuffle(self.act_buf)
    np.random.set_state(rng_state)
    np.random.shuffle(self.rew_buf)
    np.random.set_state(rng_state)
    np.random.shuffle(self.done_buf)
    self.replay_ptr = 0

  def sample_batch(self, batch_size=32):
    if self.replay_ptr + batch_size <= self.size:
      batch = dict(obs=self.obs_buf[self.replay_ptr:self.replay_ptr+batch_size],
             obs2=self.obs2_buf[self.replay_ptr:self.replay_ptr+batch_size],
             act=self.act_buf[self.replay_ptr:self.replay_ptr+batch_size],
             rew=self.rew_buf[self.replay_ptr:self.replay_ptr+batch_size],
             done=self.done_buf[self.replay_ptr:self.replay_ptr+batch_size])
      self.replay_ptr += batch_size
      return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    else:
      return None