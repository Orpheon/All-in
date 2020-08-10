import json
from trueskill import TrueSkill
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from league.agentManager import AGENT_TYPES, AgentManager

TRUESKILL_START_MU = 100
TRUESKILL_START_SIGMA = 30


class Leaderboard:

  def __init__(self, leaderboard_type, file_path):
    self.LEADERBOARD_TYPE = leaderboard_type
    self.FILE_PATH = file_path
    self.changed = True

  def load(self):
    with open(self.FILE_PATH, 'r') as f:
      data = json.load(f)
    self._ratings = data
    print('[Leaderboard <- {}]: load {}'.format(self.FILE_PATH, ' '.join('{}: {} |'.format(key, len(val))
                                                                         for key, val in self._ratings.items())))
    self.changed = False

  def save(self):
    if self.changed:
      with open(self.FILE_PATH, 'w') as f:
        json.dump(self._ratings, f)
      print('[Leaderboard -> {}]: save {}'.format(self.FILE_PATH, ' '.join('{}: {} |'.format(key, len(val))
                                                                           for key, val in self._ratings.items())))
    self.changed = False


class LeaderboardTrueskill(Leaderboard):

  def __init__(self, leaderboard_type, file_path):
    super().__init__(leaderboard_type, file_path)
    self._trueskill = TrueSkill(mu=TRUESKILL_START_MU, sigma=TRUESKILL_START_SIGMA)
    self._ratings = {'current': {}, 'history': []}

  def update_from_placings(self, placings):
    placings_ids = [agent_id for agent_id, _ in placings]
    current_ratings = [self._trueskill.Rating(*self._ratings['current'].get(agent, ())) for agent in placings_ids]
    new_rating_groups = self._trueskill.rate([(cr,) for cr in current_ratings])
    hist_append = {}
    for agent, new_rating_group in zip(placings_ids, new_rating_groups):
      new_rating = (new_rating_group[0].mu, new_rating_group[0].sigma)
      self._ratings['current'][agent] = new_rating
      hist_append[agent] = new_rating
    self._ratings['history'].append(hist_append)
    self.changed = True

  def get_all_rankings(self):
    sorted_ratings = sorted(self._ratings['current'].items(), key=lambda x: x[1][0], reverse=True)
    return sorted_ratings

  def get_rating_of_agent(self, agent_id):
    # TODO ranking != mu, but keep atm
    return self._ratings['current'].get(agent_id, (TRUESKILL_START_MU, TRUESKILL_START_SIGMA))

  def reset_rankings(self):
    self._ratings = {'current': {}, 'history': []}
    self.changed = True

  def plot_leaderboard(self, agent_manager):

    plt.figure(0)
    plt.subplot()

    mu = np.ndarray((len(self._ratings['history']),))
    sigma_2 = np.ndarray((len(self._ratings['history']),))
    x = np.arange(len(self._ratings['history']))

    agent_types = {agent_type: idx for idx, agent_type in enumerate(AGENT_TYPES.keys())}
    #palette = sns.color_palette("husl", n_colors=len(agent_types))
    palette = [(60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (220, 190, 255), (128, 0, 0), (0, 0, 128),
               (128, 128, 128), (0, 0, 0), (64, 64, 64)]
    palette = [(r / 255, g / 255, b / 255) for r, g, b in palette]
    for agent_id in self._ratings['current'].keys():
      agent_info = agent_manager.get_info(agent_id)
      color = palette[agent_types[agent_info.AGENT_TYPE]]

      for idx, game in enumerate(self._ratings['history']):
        if agent_id in game:
          mu[idx] = game[agent_id][0]
          sigma_2[idx] = game[agent_id][1] * 2
        else:
          mu[idx] = np.NaN
          sigma_2[idx] = np.NaN
      mask = np.isfinite(mu)
      sns.lineplot(x=x[mask], y=mu[mask], style=agent_info.TRAINABLE, dashes={False: [], True: [2, 2]}, color=color)
      # plt.errorbar(x=x[mask],
      #              y=mu[mask],
      #              yerr=sigma_2[mask],
      #              linestyle='-')
      # plt.ylim(87, 125)

    fake_lines = [plt.Line2D([0], [0], color=color) for color in palette]
    #plt.title('[{}]: {}'.format(self.LEADERBOARD_TYPE, self.FILE_PATH))
    plt.xlabel('rounds played')
    plt.ylabel('trueskill rating')
    plt.legend(fake_lines, agent_types)
    plt.show()

  def print_leaderboard(self, agent_manager):
    ratings = [(v[0], v[1], k, agent_manager.get_info(k))
               for k, v in self._ratings['current'].items()]
    output = "\n".join('{:>4}: {} {}.{} {:<20} {:>7.2f} +/- {:>5.2f}'
                       .format(idx + 1, agent_info.AGENT_TYPE, agent_info.ORIGIN_DIVI, agent_id, agent_info.AGENT_NAME,
                               mu, 2 * sigma)
                       for idx, (mu, sigma, agent_id, agent_info) in enumerate(sorted(ratings, reverse=True)))
    print(output)

class LeaderboardPlacingMatrix(Leaderboard):

  def __init__(self, leaderboard_type, file_path):
    super().__init__(leaderboard_type, file_path)
    self._ratings = {'current': {}, 'n_games': {}, 'agent_ids': []}

  def update_from_placings(self, placings):
    placings_ids = [agent_id for agent_id, _ in placings]

    for rank, agent_a in enumerate(placings_ids):
      for agent_b in placings_ids:
        if agent_a != agent_b:
          if agent_a not in self._ratings['agent_ids']: self._ratings['agent_ids'].append(agent_a)
          if agent_b not in self._ratings['agent_ids']: self._ratings['agent_ids'].append(agent_b)

          if agent_a not in self._ratings['n_games']: self._ratings['n_games'][agent_a] = {}
          if agent_b not in self._ratings['n_games'][agent_a]: self._ratings['n_games'][agent_a][agent_b] = 0

          if agent_a not in self._ratings['current']: self._ratings['current'][agent_a] = {}
          if agent_b not in self._ratings['current'][agent_a]: self._ratings['current'][agent_a][agent_b] = 0

          self._ratings['n_games'][agent_a][agent_b] += 1
          n_games = self._ratings['n_games'][agent_a][agent_b]
          old_r = self._ratings['current'][agent_a][agent_b]
          self._ratings['current'][agent_a][agent_b] = (rank - old_r) / n_games + old_r

    self.changed = True

  def plot_leaderboard(self, agent_manager):
    agent_ids = self._ratings['agent_ids']
    current = self._ratings['current']

    pctl = 80
    metrics = {agent_id: {
      'average': sum(current[agent_id].values()) / len(current[agent_id]),
      'median': np.median(list(current[agent_id].values())),
      'pctl': np.percentile(list(current[agent_id].values()), pctl)
    } for agent_id in agent_ids}
    ids_sorted_by_avg_rank = sorted(agent_ids, key=lambda x: metrics[x]['average'])

    n = len(agent_ids)
    tmp_arr = np.zeros((n, n))

    tmp_arr.fill(np.nan)
    data_frame = pd.DataFrame(tmp_arr)
    rename_dict = {idx: a_id for idx, a_id in enumerate(ids_sorted_by_avg_rank)}
    data_frame.rename(columns=rename_dict, index=rename_dict, inplace=True)

    for agent_a in self._ratings['current'].keys():
      for agent_b, val in self._ratings['current'][agent_a].items():
        data_frame[agent_a][agent_b] = val

    sns.set(style='white')
    cmap = sns.diverging_palette(150, 10, l=60, n=45, center="dark")
    sns.heatmap(data_frame, cmap=cmap)
    plt.title('[{}]: {}'.format(self.LEADERBOARD_TYPE, self.FILE_PATH))
    plt.xlabel('agent')
    plt.ylabel('opponent')
    plt.show()

  def print_leaderboard(self, agent_manager: AgentManager):
    agent_ids = [(agent_id, agent_manager.get_info(agent_id)) for agent_id in self._ratings['agent_ids']]
    current = self._ratings['current']

    pctl = 80

    ids_with_avg_rank = (
      (agent_id,
       agent_info,
       sum(current[agent_id].values()) / len(current[agent_id].values()),
       np.median(list(current[agent_id].values())),
       np.percentile(list(current[agent_id].values()), pctl))
      for agent_id, agent_info in agent_ids)
    ids_sorted_by_avg_rank = sorted(ids_with_avg_rank, key=lambda x: x[2])

    print('{:>4}  {} {} {:<20} {:>5} {:>5} {}pctl'.format('', 'type', 'div.id ', 'name', 'avg', 'med', pctl))
    output = "\n".join('{:>4}: {} {}.{} {:<20} {:>5.2f} {:>5.2f} {:>5.2f}'
                       .format(idx + 1, agent_info.AGENT_TYPE, agent_info.ORIGIN_DIVI, agent_id, agent_info.AGENT_NAME,
                               avg_rank, median_rank, pctl)
                       for idx, (agent_id, agent_info, avg_rank, median_rank, pctl) in
                       enumerate(ids_sorted_by_avg_rank))
    print(output)


class LeaderboardWinningsMatrix(Leaderboard):

  def __init__(self, leaderboard_type, file_path):
    super().__init__(leaderboard_type, file_path)
    self._ratings = {'current': {}, 'n_games': {}, 'agent_ids': []}
    self.pctl = 20

  def update_from_placings(self, placings):

    for agent_a, winning_a in placings:
      for agent_b, _ in placings:
        if agent_a != agent_b:
          if agent_a not in self._ratings['agent_ids']: self._ratings['agent_ids'].append(agent_a)
          if agent_b not in self._ratings['agent_ids']: self._ratings['agent_ids'].append(agent_b)

          if agent_a not in self._ratings['n_games']: self._ratings['n_games'][agent_a] = {}
          if agent_b not in self._ratings['n_games'][agent_a]: self._ratings['n_games'][agent_a][agent_b] = 0

          if agent_a not in self._ratings['current']: self._ratings['current'][agent_a] = {}
          if agent_b not in self._ratings['current'][agent_a]: self._ratings['current'][agent_a][agent_b] = 0

          self._ratings['n_games'][agent_a][agent_b] += 1
          n_games = self._ratings['n_games'][agent_a][agent_b]
          old_w = self._ratings['current'][agent_a][agent_b]
          self._ratings['current'][agent_a][agent_b] = (winning_a - old_w) / n_games + old_w

    self.changed = True

  def plot_leaderboard(self, agent_manager):
    agent_ids = self._ratings['agent_ids']
    current = self._ratings['current']
    print(sum(sum(i.values()) for i in self._ratings['n_games'].values()))


    metrics = {agent_id: {
      'average': sum(current[agent_id].values()) / len(current[agent_id]),
      'median': np.median(list(current[agent_id].values())),
      'pctl': np.percentile(list(current[agent_id].values()), self.pctl)
    } for agent_id in agent_ids}
    ids_sorted_by_avg_rank = sorted(agent_ids, key=lambda x: metrics[x]['pctl'], reverse=True)

    n = len(agent_ids)
    tmp_arr = np.zeros((n, n))

    tmp_arr.fill(np.nan)
    data_frame = pd.DataFrame(tmp_arr)
    rename_dict = {idx: a_id for idx, a_id in enumerate(ids_sorted_by_avg_rank)}
    data_frame.rename(columns=rename_dict, index=rename_dict, inplace=True)

    for agent_a in self._ratings['current'].keys():
      for agent_b, val in self._ratings['current'][agent_a].items():
        data_frame[agent_a][agent_b] = val

    sns.set(style='white')
    cmap = sns.diverging_palette(10, 150, l=60, n=45, center="dark")
    sns.heatmap(data_frame, cmap=cmap)
    plt.title('winnings matrix')
    plt.xlabel('agent')
    plt.ylabel('opponent')
    plt.show()

  def print_leaderboard(self, agent_manager: AgentManager):
    agent_ids = [(agent_id, agent_manager.get_info(agent_id)) for agent_id in self._ratings['agent_ids']]
    current = self._ratings['current']

    ids_with_avg_rank = (
      (agent_id,
       agent_info,
       sum(current[agent_id].values()) / len(current[agent_id].values()),
       np.median(list(current[agent_id].values())),
       np.percentile(list(current[agent_id].values()), self.pctl))
      for agent_id, agent_info in agent_ids)
    ids_sorted_by_avg_rank = sorted(ids_with_avg_rank, key=lambda x: x[4], reverse=True)

    print('{:>4}  {} {} {:<20} {:>7} {:>7} {}pctl'.format('', 'type', 'div.id ', 'name', 'avg', 'med', self.pctl))
    output = "\n".join('{:>4}: {} {}.{} {:<20} {:>7.2f} {:>7.2f} {:>7.2f}'
                       .format(idx + 1, agent_info.AGENT_TYPE, agent_info.ORIGIN_DIVI, agent_id, agent_info.AGENT_NAME,
                               avg_rank, median_rank, pctl)
                       for idx, (agent_id, agent_info, avg_rank, median_rank, pctl) in
                       enumerate(ids_sorted_by_avg_rank))
    print(output)

