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

    # 5 community cards x 53 (52 cards + "unknown") + 2 holecards x 52,
    # (1 position in this round + 1 folded + 1 allined + 1 pot investment total + 1 pot investment this round + 1 which player last raised) x 6
    # round x 4 (1h) + min_bet
    self.obs_dim = (5) * 53 + (2) * 52 + (1 + 1 + 1 + 1 + 1 + 1) * 6 + (1) * 5 + 1
    # Target value (mean and stddev of call ceiling, mean and stddev of raise ceiling)
    self.act_dim = 4

    self.policy = models.Policy(self.obs_dim, self.act_dim)
    self.qfn = models.Qfn(self.obs_dim)

    self.target_qfn = deepcopy(self.qfn)
    for parameter in self.target_qfn.parameters():
      parameter.requires_grad = False

    self.replaybuffer = replaybuffer.ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                                  size=self.INITAL_CAPITAL * self.N_PLAYERS * 5)

    self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config['learning_rate'])
    self.qfn_optimizer = torch.optim.Adam(self.qfn.parameters(), lr=self.config['learning_rate'])

  def act(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, allined, last_raiser, hole_cards, community_cards):
    state = self.build_network_input(player_idx, round, current_bets, min_raise, prev_round_investment, folded, allined,
                                     last_raiser, hole_cards, community_cards)

    actions = np.random.randint(0, 3, min_raise.size).astype("float")
    amounts = np.random.rand(min_raise.size) * 5

    return actions, amounts

  def round_end(self, player_idx, round, current_bets, min_raise, prev_round_investment, hole_cards, community_cards):
    pass

  def train(self):
    pass

  def spawn_executor(self):
    pass

  def build_network_input(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, allined,
                          last_raiser, hole_cards, community_cards):
    # First convert the treys card IDs into indices
    hole_cards_converted = 13 * np.log2(np.right_shift(hole_cards, 12) & 0xF) + (np.right_shift(hole_cards, 8) & 0xF)
    community_cards_converted = 13 * np.log2(np.right_shift(community_cards, 12) & 0xF) + (np.right_shift(community_cards, 8) & 0xF)
    # Then convert those indices into 1h
    hole_cards_1h = (np.arange(52) == hole_cards_converted[..., None] - 1).astype(int)
    known_community_cards_1h = (np.arange(53) == community_cards_converted[..., None] - 1).astype(int)
    # Fill missing community cards with zero
    missing_community_cards = np.zeros((self.BATCH_SIZE, 5 - community_cards.shape[1], 53))
    missing_community_cards[:, :, -1] = 1
    community_cards_1h = np.concatenate((known_community_cards_1h, missing_community_cards), axis=1)

    player_data = np.zeros((self.BATCH_SIZE, 6, self.N_PLAYERS))
    # Which player are we
    player_data[:, 0, player_idx] = 1
    # Who folded already
    player_data[:, 1, :] = folded
    # Who allin-ed already
    player_data[:, 2, :] = allined
    # Who put how much total into the pot
    player_data[:, 3, :] = (prev_round_investment + current_bets) / self.INITAL_CAPITAL
    # Who put how much this round
    player_data[:, 4, :] = (current_bets) / self.INITAL_CAPITAL
    # Who was the last to raise
    player_data[:, 5, last_raiser] = 1
    print(player_data)

    tail_data = np.zeros((self.BATCH_SIZE, 5 + 1))
    tail_data[:, round] = 1
    tail_data[:, -1] = min_raise / self.INITAL_CAPITAL

    network_input = np.concatenate((hole_cards_1h.reshape(self.BATCH_SIZE, -1),
                                    community_cards_1h.reshape(self.BATCH_SIZE, -1),
                                    player_data.reshape(self.BATCH_SIZE, -1), tail_data.reshape(self.BATCH_SIZE, -1)),
                                   axis=1)
    assert(network_input.shape[1] == self.obs_dim)

    return network_input

  def interpret_network_output(self, network_output, current_bets, prev_round_investment, min_raise):
    call_ceiling = np.random.normal(network_output[:, 0], network_output[:, 1])
    raise_ceiling = np.random.normal(network_output[:, 2], network_output[:, 3])

    actions = np.zeros(self.BATCH_SIZE)
