from agent.BaseAgent import BaseAgent
from agent.sac1 import models
from agent.sac1 import replaybuffer
from copy import deepcopy
import torch

import constants

class Sac1Agent(BaseAgent):

  def __str__(self):
    return 'SacAgent1_{0}'.format(self.agent_id)

  @classmethod
  def _config_file_path(cls):
    return './agent/sac1/config.json'

  def start_game(self, batch_size, initial_capital, n_players):
    self.BATCH_SIZE = batch_size
    self.INITAL_CAPITAL = initial_capital
    self.N_PLAYERS = n_players

    # 5 community cards + 2 holecards x 52, [1 my position + 1 folded + 1 pot investment + 1 last raise x 6 + min_bet]
    self.observation_dim = (self.BATCH_SIZE, (5 + 2 + 1), 52)
    self.action_dim = (self.BATCH_SIZE, 2)

    self.policy = models.Policy()
    self.qfn = models.Qfn()

    self.target_qfn = deepcopy(self.qfn)
    for parameter in self.target_qfn.parameters():
      parameter.requires_grad = False

    self.replaybuffer = replaybuffer.ReplayBuffer(obs_dim=self.observation_dim, act_dim=self.action_dim,
                                                  size=self.INITAL_CAPITAL * self.N_PLAYERS * 5)

    self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config['learning_rate'])
    self.qfn_optimizer = torch.optim.Adam(self.qfn.parameters(), lr=self.config['learning_rate'])


  def act(self, player_idx, round, current_bets, min_raise, prev_round_investment, hole_cards, community_cards):
    pass

  def round_end(self, player_idx, round, current_bets, min_raise, prev_round_investment, hole_cards, community_cards):
    pass

  def train(self):
    pass

  def spawnExecutor(self):
    pass
