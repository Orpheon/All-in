from agent.BaseAgent import BaseAgent
from agent.sac1 import models
from agent.sac1 import replaybuffer
from copy import deepcopy
import torch
import itertools
import numpy as np

import constants


class Sac1Agent(BaseAgent):
  def __str__(self):
    return 'SacAgent1_{0}'.format(self.agent_id)

  @classmethod
  def _config_file_path(cls):
    return './agent/sac1/config.json'

  def start_game(self, batch_size, initial_capital, n_players, alpha=0.2, gamma=0.99, polyak=0.995):
    self.BATCH_SIZE = batch_size
    self.INITAL_CAPITAL = initial_capital
    self.N_PLAYERS = n_players

    self.alpha = alpha
    self.gamma = gamma
    self.polyak = polyak

    # 5 community cards x 53 (52 cards + "unknown") + 2 holecards x 52,
    # (1 position in this round + 1 folded + 1 pot investment total + 1 pot investment this round + 1 which player last raised) x 6
    # round x 4 (1h) + min_bet
    self.obs_dim = (5) * 53 + (2) * 52 + (1 + 1 + 1 + 1 + 1) * 6 + (1) * 5 + 1
    # Action dimension (call and raise ceiling)
    self.act_dim = 2

    self.ac = models.MLPActorCritic(self.obs_dim, self.act_dim, 1)

    self.target_ac = deepcopy(self.ac)
    for parameter in self.target_ac.parameters():
      parameter.requires_grad = False

    self.replaybuffer = replaybuffer.ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                                  size=15 * self.BATCH_SIZE)

    self.pi_optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.config['learning_rate'])
    self.q_optimizer = torch.optim.Adam(itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters()),
                                        lr=self.config['learning_rate'])

    self.first_round = True
    self.prev_state = None
    self.prev_action = None

  def act(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser, hole_cards,
          community_cards):
    state = self.build_network_input(player_idx, round, current_bets, min_raise, prev_round_investment, folded,
                                     last_raiser, hole_cards, community_cards)

    network_output = self.ac.act(torch.as_tensor(state, dtype=torch.float32), deterministic=True)

    if not self.first_round:
      self.replaybuffer.store(self.prev_state, self.prev_action, 0, state, False, self.BATCH_SIZE)
    self.first_round = False
    self.prev_state = state
    self.prev_action = network_output

    actions, amounts = self.interpret_network_output(network_output, current_bets[:, player_idx],
                                                     prev_round_investment[:, player_idx], min_raise)

    return actions, amounts

  def end_trajectory(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser, hole_cards, community_cards, gains):
    state = self.build_network_input(player_idx, round, current_bets, min_raise, prev_round_investment, folded,
                                     last_raiser, hole_cards, community_cards)
    self.replaybuffer.store(self.prev_state, self.prev_action, gains / self.INITAL_CAPITAL, state, True, self.BATCH_SIZE)
    self.train()

  def train(self):
    self.replaybuffer.shuffle()
    batch = self.replaybuffer.sample_batch(batch_size=1000)
    while batch:
      self.update_parameters(batch)
      batch = self.replaybuffer.sample_batch(batch_size=1000)

  def build_network_input(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser,
                          hole_cards, community_cards):
    # First convert the treys card IDs into indices
    hole_cards_converted = 13 * np.log2(np.right_shift(hole_cards, 12) & 0xF) + (np.right_shift(hole_cards, 8) & 0xF)
    community_cards_converted = 13 * np.log2(np.right_shift(community_cards, 12) & 0xF) + (
      np.right_shift(community_cards, 8) & 0xF)
    # Then convert those indices into 1h
    hole_cards_1h = (np.arange(52) == hole_cards_converted[..., None] - 1).astype(int)
    known_community_cards_1h = (np.arange(53) == community_cards_converted[..., None] - 1).astype(int)
    # Fill missing community cards with zero
    missing_community_cards = np.zeros((self.BATCH_SIZE, 5 - community_cards.shape[1], 53))
    missing_community_cards[:, :, -1] = 1
    community_cards_1h = np.concatenate((known_community_cards_1h, missing_community_cards), axis=1)

    player_data = np.zeros((self.BATCH_SIZE, 5, self.N_PLAYERS))
    # Which player are we
    player_data[:, 0, player_idx] = 1
    # Who folded already
    player_data[:, 1, :] = folded
    # Who put how much total into the pot
    player_data[:, 2, :] = (prev_round_investment + current_bets) / self.INITAL_CAPITAL
    # Who put how much this round
    player_data[:, 3, :] = (current_bets) / self.INITAL_CAPITAL
    # Who was the last to raise
    player_data[:, 4, last_raiser] = 1

    tail_data = np.zeros((self.BATCH_SIZE, 5 + 1))
    tail_data[:, round] = 1
    tail_data[:, -1] = min_raise / self.INITAL_CAPITAL

    network_input = np.concatenate((hole_cards_1h.reshape(self.BATCH_SIZE, -1),
                                    community_cards_1h.reshape(self.BATCH_SIZE, -1),
                                    player_data.reshape(self.BATCH_SIZE, -1), tail_data.reshape(self.BATCH_SIZE, -1)),
                                   axis=1)
    assert (network_input.shape[1] == self.obs_dim)

    return network_input

  def interpret_network_output(self, network_output, current_bets, prev_round_investment, min_raise):

    actions = np.zeros(self.BATCH_SIZE, dtype=int)
    amounts = np.zeros(self.BATCH_SIZE, dtype=int)

    actions[:] = constants.FOLD

    current_stake = current_bets + prev_round_investment
    desired_raises = current_stake + min_raise < network_output[:, 0]
    desired_calls = current_stake < network_output[:, 1]

    actions[desired_raises] = constants.RAISE
    amounts[desired_raises] = network_output[desired_raises, 0] - current_stake[desired_raises]

    actions[desired_calls & np.logical_not(desired_raises)] = constants.CALL

    return actions, amounts

  # Set up function for computing SAC Q-losses
  def compute_loss_q(self, data):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

    q1 = self.ac.q1(o, a)
    q2 = self.ac.q2(o, a)

    # Bellman backup for Q functions
    with torch.no_grad():
      # Target actions come from *current* policy
      a2, logp_a2 = self.ac.pi(o2)

      # Target Q-values
      q1_pi_targ = self.target_ac.q1(o2, a2)
      q2_pi_targ = self.target_ac.q2(o2, a2)
      q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
      backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

    # MSE loss against Bellman backup
    loss_q1 = ((q1 - backup) ** 2).mean()
    loss_q2 = ((q2 - backup) ** 2).mean()
    loss_q = loss_q1 + loss_q2

    # Useful info for logging
    q_info = dict(Q1Vals=q1.detach().numpy(),
                  Q2Vals=q2.detach().numpy())

    return loss_q, q_info

  # Set up function for computing SAC pi loss
  def compute_loss_pi(self, data):
    o = data['obs']
    pi, logp_pi = self.ac.pi(o)
    q1_pi = self.ac.q1(o, pi)
    q2_pi = self.ac.q2(o, pi)
    q_pi = torch.min(q1_pi, q2_pi)

    # Entropy-regularized policy loss
    loss_pi = (self.alpha * logp_pi - q_pi).mean()

    # Useful info for logging
    pi_info = dict(LogPi=logp_pi.detach().numpy())

    return loss_pi, pi_info

  def update_parameters(self, data):
    # First run one gradient descent step for Q1 and Q2
    self.q_optimizer.zero_grad()
    loss_q, q_info = self.compute_loss_q(data)
    loss_q.backward()
    self.q_optimizer.step()

    # Record things
    # logger.store(LossQ=loss_q.item(), **q_info)

    # Freeze Q-networks so you don't waste computational effort
    # computing gradients for them during the policy learning step.
    for p in itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters()):
      p.requires_grad = False

    # Next run one gradient descent step for pi.
    self.pi_optimizer.zero_grad()
    loss_pi, pi_info = self.compute_loss_pi(data)
    loss_pi.backward()
    self.pi_optimizer.step()

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters()):
      p.requires_grad = True

    # Record things
    # logger.store(LossPi=loss_pi.item(), **pi_info)

    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
      for p, p_targ in zip(self.ac.parameters(), self.target_ac.parameters()):
        # NB: We use an in-place operations "mul_", "add_" to update target
        # params, as opposed to "mul" and "add", which would make new tensors.
        p_targ.data.mul_(self.polyak)
        p_targ.data.add_((1 - self.polyak) * p.data)