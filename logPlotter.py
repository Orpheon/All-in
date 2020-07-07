import bz2
import json
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas

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
  cashflows = []
  folding_losses = []
  winrate_per_hole_hand = {}

  def __init__(self, output_folder):
    self.output_folder = output_folder

  def plot_game_winnings(self):
    players = set()
    for game_winning in self.game_winnings:
      for p in game_winning.keys():
        players.add(p)

    for p in players:
      winnings = [gw.get(p, 0) for gw in self.game_winnings]
      plt.plot(winnings, label=p, marker='^')
    plt.legend()
    plt.title('game winnings')
    plt.xlabel('game index')
    plt.ylabel('winning')
    plt.savefig('{}game_winnings'.format(self.output_folder))
    plt.close()

  def plot_cashflow(self):
    for idx, game in enumerate(self.cashflows):
      cashflow_list = []
      for spender_idx in range(6):
        for receiver_idx in range(6):
          cashflow_list.append([game[0][spender_idx], game[0][receiver_idx], float(game[1][receiver_idx][spender_idx])])

      cashflow_data = pandas.DataFrame(np.array(cashflow_list))
      cashflow_data.columns = ['spender', 'receiver', 'amount']
      cashflow_data['spender'] = cashflow_data['spender'].astype('category')
      cashflow_data['receiver'] = cashflow_data['receiver'].astype('category')
      cashflow_data['amount'] = cashflow_data['amount'].astype('float64')
      cashflow_pivot = cashflow_data.pivot('spender', 'receiver', 'amount')

      f, ax = plt.subplots(figsize=(9, 6))
      sns.heatmap(cashflow_pivot, annot=True, fmt=".3f", linewidths=.5, ax=ax, cmap='Blues')
      plt.tight_layout()
      plt.title('game {}/{}'.format(idx + 1, len(self.cashflows)))
      plt.savefig('{}cashflow_{}-{}'.format(self.output_folder, idx + 1, len(self.cashflows)))
      plt.close()
      print('[logplotter -> {}cashflow_{}-{}]'.format(self.output_folder, idx + 1, len(self.cashflows)))

      print(cashflow_pivot.keys())
      for k1 in cashflow_pivot.keys():
        for k2 in cashflow_pivot.keys():
          if k1 < k2:
            cashflow_pivot[k1][k2], cashflow_pivot[k2][k1] = cashflow_pivot[k1][k2] - cashflow_pivot[k2][k1], cashflow_pivot[k2][k1] - cashflow_pivot[k1][k2]
      f, ax = plt.subplots(figsize=(9, 6))
      cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
      sns.heatmap(cashflow_pivot, annot=True, fmt=".3f", linewidths=.5, ax=ax, cmap=cmap)
      plt.tight_layout()
      plt.title('game {}/{}'.format(idx + 1, len(self.cashflows)))
      plt.savefig('{}norm_cashflow_{}-{}'.format(self.output_folder, idx + 1, len(self.cashflows)))
      plt.close()
      print('[logplotter -> {}norm_cashflow_{}-{}]'.format(self.output_folder, idx + 1, len(self.cashflows)))

  def plot_fold_loss(self):
    players = set()
    for game_winning in self.game_winnings:
      for p in game_winning.keys():
        players.add(p)

    for p in players:
      folding_losses = [fl.get(p, 0) for fl in self.folding_losses]
      plt.plot(folding_losses, label=p, marker='v')
    plt.legend()
    plt.title('folding losses')
    plt.xlabel('game index')
    plt.ylabel('loss')
    plt.savefig('{}folding_losses'.format(self.output_folder))
    plt.close()
    print('[logplotter -> {}folding_losses]'.format(self.output_folder))

  def load_n_most_recent_logfiles(self, n):
    file_names = [self.root_path + fn for fn in listdir(self.root_path) if
                  isfile(self.root_path + fn) and fn.endswith('bz2')]
    to_load = sorted(file_names)[-n:]
    print('{}/{} logfiles added to list'.format(n, len(file_names)))

    for idx, file_path in enumerate(to_load):
      print('loaded:', idx)
      data = self.load_logfile(file_path)
      for event in data:
        event_code = event[0]
        # print(type(event_code))
        if event_code == constants.EV_END_GAME:
          self._handle_event_ev_end_game(event[1:])

  def _handle_event_ev_end_game(self, data):
    card_values = data[0]
    winnings = data[1]
    players = data[2]
    folded = data[3]
    hole_hands = data[4]
    n_games, n_players = winnings.shape

    # winrate_per_hole_hand = {}
    # n, m = winnings.shape
    # for n_idx in range(n_games):
    #   for m_idx in range(n_players):
    #     hand = tuple(sorted(hole_hands[n_idx][m_idx]))
    #     if hand not in winrate_per_hole_hand.keys():
    #       if winnings[n_idx][m_idx] > 0:
    #         winrate_per_hole_hand[hand] = [1, 0]
    #       if winnings[n_idx][m_idx] < 0:
    #         winrate_per_hole_hand[hand] = [0, 1]
    #     else:
    #       if winnings[n_idx][m_idx] > 0:
    #         winrate_per_hole_hand[hand][0] += 1
    #       if winnings[n_idx][m_idx] < 0:
    #         winrate_per_hole_hand[hand][1] += 1
    #
    # for k, v in winrate_per_hole_hand.items():
    #   if k not in self.winrate_per_hole_hand.keys():
    #     self.winrate_per_hole_hand[k] = v
    #   else:
    #     self.winrate_per_hole_hand[k][0] += v[0]
    #     self.winrate_per_hole_hand[k][1] += v[1]

    #TODO: remove
    # return

    # winnings
    winnings_sum = np.sum(winnings, NP_VERTICAL) / n_games
    self.game_winnings.append({p: w for p, w in zip(players, winnings_sum)})

    # cashflow
    receivers = winnings > 0
    spenders = winnings < 0

    n_receivers = np.sum(receivers, NP_HORIZONTAL).reshape(len(receivers), 1)

    zero_games = []
    for i, nsp in enumerate(n_receivers):
      if nsp[0] == 0:
        zero_games.append(i)

    clean_receivers = np.delete(receivers, zero_games, axis=NP_VERTICAL)
    clean_winnings = np.delete(winnings, zero_games, axis=NP_VERTICAL)
    clean_spenders = np.delete(spenders, zero_games, axis=NP_VERTICAL)
    clean_n_receivers = np.delete(n_receivers, zero_games, axis=NP_VERTICAL)

    spendings_each = clean_spenders * clean_winnings * -1 / clean_n_receivers
    spendings_each = np.floor(spendings_each)

    cashflow_matrix = clean_receivers.transpose().dot(spendings_each)
    self.cashflows.append((players, cashflow_matrix / n_games))

    # loss
    folding_loss = np.sum(winnings * folded, NP_VERTICAL) / n_games
    self.folding_losses.append({p: fl for p, fl in zip(players, folding_loss)})

  def load_logfile(self, file_path):
    with bz2.BZ2File(file_path, 'r') as f:
      data = pickle.load(f)
    return data


if __name__ == '__main__':
  RELEVANT_LOGFILES = 1
  OUTPUT_FOLDER = 'plots/'

  logfile_collector = LogfileCollector(OUTPUT_FOLDER)
  logfile_collector.load_n_most_recent_logfiles(RELEVANT_LOGFILES)

  game_winnings = logfile_collector.game_winnings

  logfile_collector.plot_game_winnings()
  logfile_collector.plot_fold_loss()
  logfile_collector.plot_cashflow()
