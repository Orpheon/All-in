import json
from collections import namedtuple

import pandas

from agent.baseAgentNP import BaseAgentNP
from league.agentManager import AgentManager
from league.game import GameEngine
from league.logger import GenericLogger, NoneLogger

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import treys
import time

FULL_DECK = np.array(treys.Deck.GetFullDeck())

suits = {'S', 'H', 'D', 'C'}
ranks = {'2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'}


def generate_cards_best_ev():
  return [], []


def generate_random_cards():
  return [], []


def generate_sorted_deck(batch_size, n_players):
  cards = np.array([FULL_DECK for _ in range(batch_size)])
  community_cards = cards[:, :5]
  hole_cards = np.reshape(cards[:, 5:5 + 2 * n_players], (batch_size, n_players, 2))
  hole_cards = np.array([[[134224677, 134253349] for _ in range(n_players)] for _ in range(batch_size)])
  return community_cards, hole_cards


ActionReaction = namedtuple('ActionReaction', ['player_idx', 'round', 'active_rounds', 'current_bets', 'min_raise',
                                               'prev_round_investment', 'folded', 'last_raiser', 'hole_cards',
                                               'community_cards', 'action', 'amount'])


class AgentActionLogger(BaseAgentNP):

  def __init__(self, agent):
    super().__init__(None, None, None, None, None, None)
    self._agent = agent
    self.actions = []
    self.model_path = agent.MODEL_PATH

  def __str__(self):
    return 'AALogger({})'.format(str(self._agent))

  def act(self, player_idx, round, active_rounds, current_bets, min_raise, prev_round_investment, folded, last_raiser,
          hole_cards, community_cards):
    hole_cards.sort(axis=1)
    community_cards[:, 0:3].sort(axis=1)
    action, amount = self._agent.act(player_idx, round, active_rounds, current_bets, min_raise, prev_round_investment,
                                     folded, last_raiser, hole_cards, community_cards)
    self.actions.append(ActionReaction(player_idx, round, active_rounds, current_bets, min_raise, prev_round_investment,
                                       folded, last_raiser, hole_cards, community_cards, action, amount))
    return action, amount

  def initialize(self, batch_size, initial_capital, n_players):
    self._agent.initialize(batch_size, initial_capital, n_players)

  def end_trajectory(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser,
                     hole_cards, community_cards, gains):
    self._agent.end_trajectory(player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser,
                               hole_cards, community_cards, gains)

  def spawn_clone(self):
    self._agent.clone()


def load_preflop_winrates():
  with open('preflop_monte_carlo_winrates.json', 'r') as f:
    winrates = (json.loads(f.read()))
  rv = {tuple(hand): winrate for hand, winrate in winrates}
  return rv


def plot_sns_heatmap(data_pivot, title, save_to_file):
  f, ax = plt.subplots(figsize=(9, 6))
  sns.set(style="whitegrid", palette="pastel", color_codes=True)
  sns.heatmap(data_pivot, annot=True, fmt=".3f", linewidths=.5, ax=ax, cmap='Blues')
  plt.tight_layout()
  plt.title(title)
  if save_to_file:
    plt.savefig(title)
  else:
    plt.show()
  plt.close()


def plot_curves(x, ys, lbls, title, save_to_file):
  # TODO: remove prints
  x = np.array(x)
  ys = np.array(ys)
  for idx in range(len(ys)):
    plt.bar(x, ys[idx, :], bottom=np.sum(ys[:idx, :], axis=0))
  plt.title(title)
  plt.xlabel('preflop handvalue (win/loss)')
  plt.ylabel('frequency (percent)')
  plt.legend(lbls)
  if save_to_file:
    plt.savefig(title)
  else:
    plt.show()
  plt.close()


def plot_hand_strength_bet_distribution(agent_action_logger: AgentActionLogger):
  winrates = load_preflop_winrates()

  action_tables = {}
  amount_tables = {}
  for i in agent_action_logger.actions:
    for j in range(N_TESTCASES):
      hand_strength = winrates[tuple(sorted(i.hole_cards[j]))]
      if hand_strength not in action_tables.keys():
        action_tables[hand_strength] = {i.action[j]: 1}
        amount_tables[hand_strength] = {i.amount[j]: 1}
      else:
        action_tables[hand_strength][i.action[j]] = action_tables[hand_strength].get(i.action[j], 0) + 1
        amount_tables[hand_strength][i.amount[j]] = amount_tables[hand_strength].get(i.amount[j], 0) + 1

  available_actions = [0, 1, 2]
  available_amounts = set()
  for a in amount_tables.values():
    for k in a.keys():
      available_amounts.add(k)
  hand_strengths_x = sorted(list(action_tables.keys()))
  action_ys = [[action_tables[hs].get(aa, 0) / sum(action_tables[hs].values())
                for hs in hand_strengths_x]
               for aa in available_actions]
  amount_ys = [[amount_tables[hs].get(aa, 0) / sum(amount_tables[hs].values())
                for hs in hand_strengths_x]
               for aa in available_amounts]

  print('-' * 50)
  for i, e in zip(available_actions, [[action_tables[hs].get(aa, 0) for hs in hand_strengths_x]
                                      for aa in available_actions]):
    print(i, sum(e))
  print('-' * 50)
  for i, e in sorted(zip(available_amounts, [[amount_tables[hs].get(aa, 0) for hs in hand_strengths_x]
                                      for aa in available_amounts]), key=lambda x: x[0]):
    print(i, sum(e))
  print('-' * 50)

  p_name = agent_action_logger._agent.AGENT_NAME
  plot_curves(hand_strengths_x, action_ys, available_actions, 'FE_action_per_hs {}'.format(p_name), False)
  plot_curves(hand_strengths_x, amount_ys, available_amounts, 'FE_amount_per_hs {}'.format(p_name), False)

  # TODO: remove return
  return

  actions = {}
  amounts = {}
  for i in agent_action_logger.actions:
    for j in range(N_TESTCASES):
      hand_strength = winrates[tuple(sorted(i.hole_cards[j]))]
      actions[hand_strength, i.action[j]] = actions.get((hand_strength, i.action[j]), 0) + 1
      amounts[hand_strength, i.amount[j]] = amounts.get((hand_strength, i.amount[j]), 0) + 1

  hs_action = np.array([[key[0], key[1], val] for key, val in actions.items()])
  hs_amount = np.array([[key[0], key[1], val] for key, val in amounts.items()])

  hs_action_data = pandas.DataFrame(hs_action)
  hs_amount_data = pandas.DataFrame(hs_amount)

  hs_action_data.columns = ['hand_strength', 'action', 'frequency']
  hs_amount_data.columns = ['hand_strength', 'amount', 'frequency']

  hs_action_pivot = hs_action_data.pivot('action', 'hand_strength', 'frequency')
  hs_amount_pivot = hs_amount_data.pivot('amount', 'hand_strength', 'frequency')

  plot_sns_heatmap(data_pivot=hs_action_pivot, title='fixed_environment_action_per_hs', save_to_file=False)
  plot_sns_heatmap(data_pivot=hs_amount_pivot, title='fixed_environment_amount_per_hs', save_to_file=False)


def print_winnings(winnings):
  print('-' * 100)
  print('winnings:')
  print(
    '{:^30}: {:^18} | {:^8} {:^8} {:^8} {:^8} {:^8} {:^8}'.format('name', 'mean winning', 'SB', 'BB', 'EP', 'MP', 'CO',
                                                                  'BTN'))
  print('-' * 100)
  for p, sw in sorted(winnings.items(), key=lambda x: sum(x[1]), reverse=True):
    print('{:<30}: {:<18.3f} | {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:8.2f}'
          .format(str(p), sum(sw) / 6, sw[0], sw[1], sw[2], sw[3], sw[4], sw[5]))
  print('-' * 100)


if __name__ == '__main__':
  # config
  N_TESTCASES = 10_000
  INITIAL_CAPITAL = 200
  SMALL_BLIND = 1
  BIG_BLIND = 2
  N_PLAYERS = 6

  N_RUNS = 1

  print('started with following config:')
  print('N_TESTCASES:    ', N_TESTCASES)
  print('INITIAL_CAPITAL:', INITIAL_CAPITAL)
  print('SMALL_BLIND:    ', SMALL_BLIND)
  print('BIG_BLIND:      ', BIG_BLIND)
  print('N_PLAYERS       ', N_PLAYERS)
  print('N_RUNS:         ', N_RUNS)

  agent_manager = AgentManager(file_path='./savefiles/agent_manager.json', models_path='./models',
                               possible_agent_names=None)
  agent_manager.load()

  logger = GenericLogger()
  # logger = NoneLogger()

  # community_cards, hole_hands = generate_sorted_deck(N_TESTCASES, N_PLAYERS)
  # mock_game_engine = MockGameEngine(N_TESTCASES, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND, community_cards, hole_hands,
  #                                  logger)

  mock_game_engine = GameEngine(N_TESTCASES, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND, logger)

  # agent_ids = ['1701'] + ['2034', '2034', '2034', '5223', '5223']
  agent_ids = ['2034', '2034', '2034', '5223', '5223', '5223']
  players = [AgentActionLogger(agent_manager.get_instance(agent_id)) for agent_id in agent_ids]

  for p in players: print(p)

  cumulated_winnings = {str(p): [-1 for _ in range(6)] for p in players}

  for _ in range(N_RUNS):
    for seat_shift in range(0, 6):
      t1 = time.time()
      total_winnings = mock_game_engine.run_game(players)
      t2 = time.time()
      print("Time required: {}s".format(t2 - t1))
      winnings = np.sum(total_winnings, axis=0).tolist()

      for s, (p, w) in enumerate(zip(players, winnings)):
        cumulated_winnings[str(p)][s] = w / N_TESTCASES
      players = players[-1:] + players[:-1]

  print_winnings(cumulated_winnings)

  for p in players:
    plot_hand_strength_bet_distribution(p)
