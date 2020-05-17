import bz2
import json
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas

from pokereval.card import Card
from pokereval.hand_evaluator import HandEvaluator

from os import listdir
from os.path import isfile, join

import constants

cards = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
rank_map = {name: idx + 2 for idx, name in enumerate(cards)}
suit_map = {'s': 1, 'h': 2, 'd': 3, 'c': 4}

NP_VERTICAL = 0
NP_HORIZONTAL = 1


class LogfileCollector:

  root_path = './gamelogs/'
  game_winnings = []

  def load_n_most_recent_logfiles(self, n):
    file_names = [self.root_path + fn for fn in listdir(self.root_path) if isfile(self.root_path + fn) and fn.endswith('bz2')]
    to_load = sorted(file_names)[-n:]
    print('{}/{} logfiles added to list'.format(n, len(file_names)))

    for file_path in to_load:
      data = self.load_logfile(file_path)
      for event in data:
        event_code = event[0]
        print(type(event_code))
        if event_code == constants.EV_END_GAME:
          self.handle_event_ev_end_game(event[1:])

  def handle_event_ev_end_game(self, data):
    #card_values = data[0]
    winnings = data[1]
    players = data[2]
    winnings_sum = np.sum(winnings, NP_VERTICAL)
    self.game_winnings.append({p: w for p, w in zip(players, winnings_sum)})

  def load_logfile(self, file_path):
    with bz2.BZ2File(file_path, 'r') as f:
      data = pickle.load(f)
    return data


class LogPlotter:
  path = 'logs/evaluation_2020-04-09_182218.json'

  def load_file(self):
    with open(self.path, 'r') as f:
      file_content = f.read()
      self._data = json.loads(file_content)

  def show_bet_frequencies(self, data: pandas.DataFrame):
    pivot = data.pivot('bet_amount', 'card_value', 'frequency')
    sns.relplot(x="card_value", y="bet_amount", size="frequency",
                alpha=.5, palette="muted", data=data)
    plt.title(player)
    plt.show()
    plt.close()


def cardval_from_str(hole):
  cards = [Card(rank_map[card[0]], suit_map[card[1]]) for card in hole]
  return HandEvaluator.evaluate_hand(cards[0:2], cards[2:5])


if __name__ == '__main__':

  logfile_collector = LogfileCollector()
  logfile_collector.load_n_most_recent_logfiles(5)

  game_winnings = logfile_collector.game_winnings
  players = set()
  for game_winning in game_winnings:
    for p in game_winning.keys():
      players.add(p)

  for p in players:
    winnings = [gw.get(p, 0) for gw in game_winnings]
    plt.plot(winnings, label=p)
  plt.legend()
  plt.show()

  #TODO:
  exit()

  lp = LogPlotter()
  lp.load_file()

  all_players = {seat['uuid']: seat['name'] for seat in lp._data['round_1']['round_state']['seats']}
  action_data = {player: {} for player in all_players.values()}
  cashflow = {name_from: {name_to: 0 for name_to in all_players.values()} for name_from in all_players.values()}

  # collect data
  for round in lp._data.values():
    seat_holecard = {seat['uuid']: seat['hole_card'] for seat in round['round_state']['seats']}

    losses = {seat['name']: 200 - seat['stack'] for seat in round['round_state']['seats']}
    spenders = [name for name, loss in losses.items() if loss > 0]
    receivers = [name for name, loss in losses.items() if loss < 0]

    for spender in spenders:
      for receiver in receivers:
        cashflow[spender][receiver] += losses[spender] / len(receivers)

    for _, round in round['round_state']['action_histories'].items():
      for action in round:
        if action['action'] != 'SMALLBLIND' and action['action'] != 'BIGBLIND':
          card_val = cardval_from_str(seat_holecard[action['uuid']])
          bet_amount = action.get('amount', 0)
          p_name = all_players[action['uuid']]
          action_data[p_name][(card_val, bet_amount)] = action_data[p_name].get((card_val, bet_amount), 0) + 1

  # transform data
  cashflow_list = []
  for spender, receivers in cashflow.items():
    for receiver, amount in receivers.items():
      cashflow_list.append([spender, receiver, int(amount)])

  cashflow_data = pandas.DataFrame(np.array(cashflow_list))
  cashflow_data.columns = ['spender', 'receiver', 'amount']
  cashflow_data['spender'] = cashflow_data['spender'].astype('category')
  cashflow_data['receiver'] = cashflow_data['receiver'].astype('category')
  cashflow_data['amount'] = cashflow_data['amount'].astype('int64')

  cashflow_pivot = cashflow_data.pivot('spender', 'receiver', 'amount')

  f, ax = plt.subplots(figsize=(9, 6))
  sns.heatmap(cashflow_pivot, annot=True, fmt="d", linewidths=.5, ax=ax)
  plt.tight_layout()
  plt.show()

  for player, actions in action_data.items():

    bet_amount_hist = []
    for act,freq in actions.items():
      for _ in range(freq):
        bet_amount_hist.append(act[1])
    plt.hist(bet_amount_hist, bins=100)
    plt.title(player)
    plt.show()

    data_array = np.array([[act[0], act[1], freq] for act, freq in actions.items()])

    data = pandas.DataFrame(data_array)
    data.columns = ['card_value', 'bet_amount', 'frequency']

    lp.show_bet_frequencies(data)
