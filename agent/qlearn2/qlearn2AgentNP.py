from agent.baseAgentNP import BaseAgentNP
from agent.qlearn2 import models
from agent.qlearn2 import replaybuffer
from league.spinningupLogger import EpochLogger

import torch
import os
import numpy as np

import constants

DEVICE = 'cuda'
NOISE_LEVEL = 0.1
Q_LEARNING_RATE = 0.001
ROOT_PATH = 'qlearn2'
REPLAYBUFFER_SIZE = 60

class Qlearn2AgentNP(BaseAgentNP):
  MODEL_FILES = ['q.modelb']
  logger = EpochLogger(output_dir='qlearn2/logs', output_fname='progress.csv')

  def __str__(self):
    return 'Qln2 {} {}'.format('T' if self.trainable else 'N', super().__str__())

  def initialize(self, batch_size, initial_capital, n_players):
    self.BATCH_SIZE = batch_size
    self.INITAL_CAPITAL = initial_capital
    self.N_PLAYERS = n_players

    # 5 community cards x 53 (52 cards + "unknown") + 2 holecards x 52,
    # (1 position in this round + 1 folded + 1 pot investment total + 1 pot investment this round + 1 which player last raised) x 6
    # round x 4 (1h)
    self.obs_dim = (5) * 53 + (2) * 52 + (1 + 1 + 1 + 1 + 1) * 6 + (1) * 5
    self.act_dim = 7

    self.q = models.MLPQFunction(self.obs_dim, self.act_dim, trainable=self.trainable, device=DEVICE)

    self.possible_actions = torch.zeros((self.BATCH_SIZE, self.act_dim), device=DEVICE)
    self.possible_raises = np.array([0, 0, 4, 10, 20, 100, 200])

    if self.trainable:
      self.reward = torch.zeros(self.BATCH_SIZE)
      self.replaybuffer = replaybuffer.ReplayBuffer(obs_dim=self.obs_dim,
                                                    act_dim=self.act_dim,
                                                    batch_size=self.BATCH_SIZE,
                                                    size=REPLAYBUFFER_SIZE,
                                                    device=DEVICE)

      self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=Q_LEARNING_RATE)

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

    actions, amounts, actions_serialized = self.choose_action(
      torch.as_tensor(state, dtype=torch.float32, device=DEVICE),
      current_bets, prev_round_investment[:, player_idx] + current_bets[:, player_idx]
    )

    if self.trainable:
      if not self.first_round:
        self.replaybuffer.store(obs=self.prev_state,
                                act=self.prev_action,
                                next_obs=state,
                                active_games=active_games)
      self.first_round = False
      self.prev_state = state
      self.prev_action = actions_serialized

    return actions, amounts

  def end_trajectory(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser,
                     hole_cards, community_cards, gains):

    #TODO: bugfix to prevent crash in case that agent never acted before game finish
    if self.trainable and self.prev_state is not None:
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
                              active_games=np.ones(self.BATCH_SIZE))
      self.logger.store(Reward=scaled_gains, LostInFolding=lost_money, LostGeneral=(gains / self.INITAL_CAPITAL))
      self.train()

      self.save_model()
      # FIXME: Remember that replaybuffer is *not* emptied here

  def train(self):
    state = self.replaybuffer.sample_state()

    while state:
      self.update_parameters(state)
      state = self.replaybuffer.sample_state()

    self.log_everything()

  def log_everything(self):
    self.logger.log_tabular('Folds', average_only=True)
    self.logger.log_tabular('Calls', average_only=True)
    for i in range(2, self.act_dim):
      self.logger.log_tabular('Raises ' + str(self.possible_raises[i]), average_only=True)
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

  def choose_action(self, network_input, current_bets, total_investment):
    scores = np.ndarray((self.BATCH_SIZE, self.act_dim))
    investment_normalized = (-total_investment / self.INITAL_CAPITAL - (self.N_PLAYERS / 2 - 1)) * 2 / self.N_PLAYERS
    with torch.no_grad():
      scores[:, 0] = investment_normalized
      for idx in range(1, self.possible_actions.shape[1]):
        onehot_actions = torch.eye(self.act_dim, device=DEVICE)[
          torch.full((self.BATCH_SIZE,), idx, dtype=torch.long, device=DEVICE)
        ]
        scores[:, idx] = self.q(network_input, onehot_actions).cpu().numpy()

      actions = np.argmax(scores, axis=1)

    if self.trainable:
      dice = np.random.random(self.BATCH_SIZE)
      rand_actions = np.random.randint(1, self.act_dim, self.BATCH_SIZE)
      actions[dice <= NOISE_LEVEL] = rand_actions[dice <= NOISE_LEVEL]

    actions_array = np.eye(self.act_dim)[actions]
    amounts = self.possible_raises[actions]

    actions[actions > constants.RAISE] = constants.RAISE

    actions[current_bets.sum(axis=1) == 0] = constants.CALL

    self.logger.store(Calls=100 * np.mean(actions == constants.CALL),
                      Folds=100 * np.mean(actions == constants.FOLD))
    for i in range(2, self.act_dim):
      self.logger.store(**{"Raises " + str(self.possible_raises[i]): 100 * np.mean(actions_array[:, i])})

    return actions, amounts, actions_array

  # Set up function for computing Q-losses
  def compute_loss_q(self, data):
    o, a, o2, active = data['obs'], data['act'], data['obs2'], data['active']

    q = self.q(o, a)

    loss_q = (active * (q - self.reward) ** 2).sum()

    # Useful info for logging
    q_info = dict(QVals=(q).cpu().detach().numpy())
    # q_info = {}

    return loss_q, q_info

  def update_parameters(self, data):
    # First run one gradient descent step for Q1 and Q2
    self.q_optimizer.zero_grad()
    loss_q, q_info = self.compute_loss_q(data)
    loss_q.backward()
    self.q_optimizer.step()

    # Record things
    self.logger.store(LossQ=loss_q.item(), **q_info)

  def load_model(self):
    if os.path.exists(self.model_path):
      self.q.load(self.model_path)
      if self.trainable:
        self.q_optimizer.load_state_dict(torch.load(os.path.join(self.model_path, 'q_opt.optb')))

  def save_model(self):
    print('saved', self.model_path)
    self.q.save(self.model_path)
    if self.trainable:
      torch.save(self.q_optimizer.state_dict(), os.path.join(self.model_path, 'q_opt.optb'))
