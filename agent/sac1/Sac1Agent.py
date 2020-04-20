from agent.BaseAgent import BaseAgent
from agent.sac1 import models
from agent.sac1 import replaybuffer
from copy import deepcopy
import torch
import treys
import numpy as np

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
    self.obs_dim = (5 + 2) * 52 + (1 + 1 + 1 + 1) * 6 + 1
    self.act_dim = 2

    self.policy = models.Policy(self.obs_dim, self.act_dim)
    self.qfn = models.Qfn(self.obs_dim)

    self.target_qfn = deepcopy(self.qfn)
    for parameter in self.target_qfn.parameters():
      parameter.requires_grad = False

    self.replaybuffer = replaybuffer.ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                                  size=self.INITAL_CAPITAL * self.N_PLAYERS * 5)

    self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config['learning_rate'])
    self.qfn_optimizer = torch.optim.Adam(self.qfn.parameters(), lr=self.config['learning_rate'])

  def act(self, player_idx, round, current_bets, min_raise, prev_round_investment, hole_cards, community_cards):
    state = self.build_network_input(player_idx, round, current_bets, min_raise, prev_round_investment, hole_cards,
                                     community_cards)

    actions = np.random.randint(0, 3, min_raise.size).astype("float")
    amounts = np.random.rand(min_raise.size) * 5

    return actions, amounts

  def round_end(self, player_idx, round, current_bets, min_raise, prev_round_investment, hole_cards, community_cards):
    pass

  def train(self):
    pass

  def spawn_executor(self):
    pass

  def build_network_input(self, player_idx, round, current_bets, min_raise, prev_round_investment, hole_cards,
                          community_cards):
    # First convert the treys card IDs into indices
    hole_cards_converted = 13 * np.log2(np.right_shift(hole_cards, 12) & 0xF) + (np.right_shift(hole_cards, 8) & 0xF)
    community_cards_converted = 13 * np.log2(np.right_shift(community_cards, 12) & 0xF) + (np.right_shift(community_cards, 8) & 0xF)
    # Then convert those indices into 1h
    hole_cards_1h = (np.arange(52) == hole_cards_converted[..., None] - 1).astype(int)
    community_cards_1h = (np.arange(52) == community_cards_converted[..., None] - 1).astype(int)

    # retwork_input = torch.zeros(self.observation_dim)
    # network_input[:, :2, :]
    #
    #
    # print(hole_cards_1h.shape, community_cards_1h.shape)
    # input()
    return 0
