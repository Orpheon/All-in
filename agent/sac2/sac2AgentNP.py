from agent.baseAgentNP import BaseAgentNP
from agent.sac2 import models
from agent.sac2 import replaybuffer
from league.spinningupLogger import EpochLogger

import torch
import itertools
import os
import numpy as np

import constants

ALPHA = 0.005
DEVICE = 'cuda'
GAMMA = 1.0
PI_LEARNING_RATE = 0.1
Q_LEARNING_RATE = 0.001
ROOT_PATH = 'sac1'
REPLAYBUFFER_SIZE = 30


class Sac2AgentNP(BaseAgentNP):
  MODEL_FILES = ['policy.modelb', 'q1.modelb', 'q2.modelb']
  logger = EpochLogger(output_dir='sac2/logs', output_fname='progress.csv')

  def initialize(self, batch_size, initial_capital, n_players):
    self.BATCH_SIZE = batch_size
    self.REPLAY_BATCH_SIZE = 1000
    self.INITAL_CAPITAL = initial_capital
    self.N_PLAYERS = n_players

    # 5 community cards x 53 (52 cards + "unknown") + 2 holecards x 52,
    # (1 position in this round + 1 folded + 1 pot investment total + 1 pot investment this round + 1 which player last raised) x 6
    # round x 4 (1h)
    self.obs_dim = (5) * 53 + (2) * 52 + (1 + 1 + 1 + 1 + 1) * 6 + (1) * 5
    # Action dimensions
    self.act_dim = 4

    self.ac = models.MLPActorCritic(self.obs_dim, self.act_dim, 1, trainable=self.TRAINABLE, device=DEVICE)

    if self.TRAINABLE:
      self.reward = torch.zeros(self.BATCH_SIZE)
      self.replaybuffer = replaybuffer.ReplayBuffer(obs_dim=self.obs_dim,
                                                    act_dim=self.act_dim,
                                                    size=REPLAYBUFFER_SIZE * self.BATCH_SIZE,
                                                    device=DEVICE)

      self.pi_optimizer = torch.optim.Adam(self.ac.parameters(), lr=PI_LEARNING_RATE)
      self.q_optimizer = torch.optim.Adam(itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters()),
                                          lr=Q_LEARNING_RATE)

      self.first_round = True
      self.prev_state = None
      self.prev_action = None

    self.load_model()

  def act(self, player_idx, round, active_games, current_bets, min_raise, prev_round_investment, folded, last_raiser,
          hole_cards, community_cards):
    state = self.build_network_input(player_idx, round, current_bets, min_raise, prev_round_investment, folded,
                                     last_raiser, hole_cards, community_cards)
    hole_cards.sort(axis=1)
    community_cards[:, 0:3].sort(axis=1)

    network_output = self.ac.act(torch.as_tensor(state, dtype=torch.float32),
                                 deterministic=not self.TRAINABLE)

    if self.TRAINABLE:
      if not self.first_round:
        self.replaybuffer.store(obs=self.prev_state[active_games],
                                act=self.prev_action[active_games],
                                next_obs=state[active_games],
                                indices=np.arange(self.BATCH_SIZE)[active_games])
      self.first_round = False
      self.prev_state = state
      self.prev_action = network_output

    actions, amounts = self.interpret_network_output(network_output,
                                                     current_bets,
                                                     prev_round_investment,
                                                     player_idx,
                                                     min_raise)

    return actions, amounts

  def end_trajectory(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser,
                     hole_cards, community_cards, gains):
    # TODO: bugfix to prevent crash in case that agent never acted before game finish
    if self.TRAINABLE and self.prev_state is not None:
      state = self.build_network_input(player_idx, round, current_bets, min_raise, prev_round_investment, folded,
                                       last_raiser, hole_cards, community_cards)
      scaled_gains = (gains / self.INITAL_CAPITAL - (self.N_PLAYERS / 2 - 1)) * 2 / self.N_PLAYERS
      # DEBUGTOOL
      lost_money = (gains / self.INITAL_CAPITAL)
      lost_money[folded[:, player_idx] == 0] = 0

      self.reward = torch.Tensor(scaled_gains).to(DEVICE)
      self.replaybuffer.store(obs=self.prev_state,
                              act=self.prev_action,
                              next_obs=state,
                              indices=np.arange(self.BATCH_SIZE))
      self.logger.store(Reward=scaled_gains, LostInFolding=lost_money, LostGeneral=(gains / self.INITAL_CAPITAL))
      self.train()

      self.save_model()
      # FIXME: Remember that replaybuffer is *not* emptied here

  def train(self):
    self.replaybuffer.shuffle()
    batch = self.replaybuffer.sample_batch(batch_size=min(self.REPLAY_BATCH_SIZE, self.BATCH_SIZE))

    while batch:
      self.update_parameters(batch)
      batch = self.replaybuffer.sample_batch(batch_size=min(self.REPLAY_BATCH_SIZE, self.BATCH_SIZE))

    self.log_everything()

  def log_everything(self):
    self.logger.log_tabular('QContribPiLoss', with_min_and_max=True, average_only=True)
    self.logger.log_tabular('LossPi', average_only=True)
    self.logger.log_tabular('EntropyBonus', average_only=True)
    self.logger.log_tabular('Raises', average_only=True)
    self.logger.log_tabular('Calls', average_only=True)
    self.logger.log_tabular('Folds', average_only=True)
    self.logger.log_tabular('LostInFolding', with_min_and_max=True, average_only=True)
    self.logger.log_tabular('LostGeneral', with_min_and_max=True, average_only=True)
    self.logger.log_tabular('QVals', with_min_and_max=True, average_only=True)
    self.logger.log_tabular('Reward', average_only=True)
    self.logger.log_tabular('LossQ', average_only=True)
    self.logger.dump_tabular()

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
    # Have a 53rd column in the 1h to indicate missing cards, and fill that with ones where relevant
    missing_community_cards[:, :, -1] = 1
    community_cards_1h = np.concatenate((known_community_cards_1h, missing_community_cards), axis=1)

    player_data = np.zeros((self.BATCH_SIZE, 5, self.N_PLAYERS))
    # Who folded already
    player_data[:, 0, :] = folded
    # Who put how much total into the pot
    player_data[:, 1, :] = (prev_round_investment + current_bets) / self.INITAL_CAPITAL
    # Who put how much this round
    player_data[:, 2, :] = (current_bets) / self.INITAL_CAPITAL
    # Who was the last to raise
    player_data[:, 3, :] = np.eye(self.N_PLAYERS)[last_raiser]
    # Reorder the first four to correspond to player_idx
    player_data = np.concatenate((player_data[:, :, player_idx:], player_data[:, :, :player_idx]), axis=2)
    # Which player are we
    player_data[:, 4, player_idx] = 1

    tail_data = np.zeros((self.BATCH_SIZE, 5))
    tail_data[:, round] = 1

    network_input = np.concatenate((hole_cards_1h.reshape(self.BATCH_SIZE, -1),
                                    community_cards_1h.reshape(self.BATCH_SIZE, -1),
                                    player_data.reshape(self.BATCH_SIZE, -1), tail_data.reshape(self.BATCH_SIZE, -1)),
                                   axis=1)

    assert (network_input.shape[1] == self.obs_dim)

    return network_input

  def interpret_network_output(self, network_output, current_bets, prev_round_investment, player_idx, min_raise):

    chosen_action = np.argmax(network_output[:, :3], axis=1)
    actions = np.array([constants.FOLD, constants.CALL, constants.RAISE])[chosen_action]

    actions[current_bets.sum(axis=1) == 0] = constants.CALL

    current_stake = current_bets[:, player_idx] + prev_round_investment[:, player_idx]
    amounts = np.clip((network_output[:, 1] + 1) * self.INITAL_CAPITAL / 2, min_raise,
                      self.INITAL_CAPITAL - current_stake)

    self.logger.store(Raises=100 * np.mean(actions == constants.RAISE),
                      Calls=100 * np.mean(actions == constants.CALL),
                      Folds=100 * np.mean(actions == constants.FOLD),
                      )

    return actions, amounts

  # Set up function for computing SAC Q-losses
  def compute_loss_q(self, data):
    o, a, o2, idx = data['obs'], data['act'], data['obs2'], data['idx']

    q1 = self.ac.q1(o, a)
    q2 = self.ac.q2(o, a)

    with torch.no_grad():
      # Target actions come from *current* policy
      a2, logp_a2 = self.ac.pi(o2)
      target = self.reward[idx] - ALPHA * logp_a2

    # MSE loss against Bellman backup
    loss_q1 = ((q1 - target) ** 2).mean()
    loss_q2 = ((q2 - target) ** 2).mean()
    loss_q = loss_q1 + loss_q2

    # Useful info for logging
    q_info = dict(QVals=((q1 + q2) / 2).cpu().detach().numpy())
    # q_info = {}

    return loss_q, q_info

  # Set up function for computing SAC pi loss
  def compute_loss_pi(self, data):
    o = data['obs']
    action, logp_pi = self.ac.pi(o)
    q1_pi = self.ac.q1(o, action)
    q2_pi = self.ac.q2(o, action)
    q_pi = torch.min(q1_pi, q2_pi)

    # Entropy-regularized policy loss
    loss_pi = (ALPHA * logp_pi - q_pi).mean()
    self.logger.store(
      QContribPiLoss=(torch.abs(q_pi) / (torch.abs(q_pi) + torch.abs(ALPHA * logp_pi))).mean().cpu().detach().numpy())

    # Useful info for logging
    pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())
    # pi_info = {}

    return loss_pi, pi_info

  def update_parameters(self, data):
    # First run one gradient descent step for Q1 and Q2
    self.q_optimizer.zero_grad()
    loss_q, q_info = self.compute_loss_q(data)
    loss_q.backward()
    self.q_optimizer.step()

    # Record things
    self.logger.store(LossQ=loss_q.item(), **q_info)

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
    self.logger.store(LossPi=loss_pi.item(), **pi_info)

  def load_model(self):
    if os.path.exists(self.MODEL_PATH):
      self.ac.load(self.MODEL_PATH)
      if self.TRAINABLE:
        self.pi_optimizer.load_state_dict(torch.load(os.path.join(self.MODEL_PATH, 'pi_opt.optb')))
        self.q_optimizer.load_state_dict(torch.load(os.path.join(self.MODEL_PATH, 'q_opt.optb')))

  def save_model(self):
    print('saved', self.MODEL_PATH)
    self.ac.save(self.MODEL_PATH)
    if self.TRAINABLE:
      torch.save(self.pi_optimizer.state_dict(), os.path.join(self.MODEL_PATH, 'pi_opt.optb'))
      torch.save(self.q_optimizer.state_dict(), os.path.join(self.MODEL_PATH, 'q_opt.optb'))
