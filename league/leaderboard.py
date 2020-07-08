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

  def __init__(self, file_path):
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

  def __init__(self, file_path):
    super().__init__(file_path)
    self._trueskill = TrueSkill(mu=TRUESKILL_START_MU, sigma=TRUESKILL_START_SIGMA)
    self._ratings = {'current': {}, 'history': []}

  def update_from_placings(self, agent_ids):
    current_ratings = [self._trueskill.Rating(*self._ratings['current'].get(agent, ())) for agent in agent_ids]
    new_rating_groups = self._trueskill.rate([(cr,) for cr in current_ratings])
    hist_append = {}
    for agent, new_rating_group in zip(agent_ids, new_rating_groups):
      new_rating = (new_rating_group[0].mu, new_rating_group[0].sigma)
      self._ratings['current'][agent] = new_rating
      hist_append[agent] = new_rating
    self._ratings['history'].append(hist_append)
    self.changed = True

  def get_all_rankings(self):
    sorted_ratings = sorted(self._ratings['current'].items(), key=lambda x: x[1][0], reverse=True)
    return sorted_ratings

  def get_ranking(self, agent_id):
    # TODO ranking != mu, but keep atm
    return self._ratings['current'].get(agent_id, (TRUESKILL_START_MU, TRUESKILL_START_SIGMA))

  def reset_rankings(self):
    self._ratings = {'current': {}, 'history': []}
    self.changed = True

  def plot_history(self, agent_manager):

    plt.figure(0)
    plt.subplot()

    mu = np.ndarray((len(self._ratings['history']),))
    sigma_2 = np.ndarray((len(self._ratings['history']),))
    x = np.arange(len(self._ratings['history']))

    agent_types = {agent_type: idx for idx, agent_type in enumerate(AGENT_TYPES.keys())}
    # palette = sns.color_palette("husl", n_colors=len(agent_types))
    palette = [(60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (220, 190, 255), (128, 0, 0), (0, 0, 128),
               (128, 128, 128), (0, 0, 0)]
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
    plt.legend(fake_lines, agent_types)
    plt.show()

  def print_current_ratings(self, agent_manager):
    ratings = [(v[0], v[1], k, agent_manager.get_info(k).AGENT_TYPE,
                'S' if agent_manager.get_info(k).TRAINABLE else 'T')
               for k, v in self._ratings['current'].items()]
    output = "\n".join("{:>3}: {:>10} {} {}: {:>7.2f} +/- {:>5.2f}".format(idx + 1, at, aid, tr, rating, 2 * sd)
                       for idx, (rating, sd, aid, at, tr) in enumerate(sorted(ratings, reverse=True)))
    print(output)


class LeaderboardPlacingMatrix(Leaderboard):

  def __init__(self, file_path):
    super().__init__(file_path)
    self._ratings = {'current': {}, 'n_games': {}, 'agent_ids': []}

  def update_from_placings(self, agent_ids):

    for rank, agent_a in enumerate(agent_ids):
      for agent_b in agent_ids:
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

  def plot_matrix(self):
    ids = self._ratings['agent_ids']

    current = self._ratings['current']
    ids_sorted_by_avg_rank = sorted(ids, key=lambda x: sum(current[x].values()) / len(current[x].values()))

    n = len(ids)
    tmp_arr = np.zeros((n, n))

    tmp_arr.fill(np.nan)
    data_frame = pd.DataFrame(tmp_arr)
    rename_dict = {idx: a_id for idx, a_id in enumerate(ids_sorted_by_avg_rank)}
    data_frame.rename(columns=rename_dict, index=rename_dict, inplace=True)

    for agent_a in self._ratings['current'].keys():
      for agent_b, val in self._ratings['current'][agent_a].items():
        data_frame[agent_a][agent_b] = val

    sns.set(style='white')
    cmap = sns.color_palette("coolwarm", 15)
    sns.heatmap(data_frame, cmap=cmap)
    plt.xlabel('agent')
    plt.ylabel('opponent')
    plt.show()

  def print_leaderboard(self, agent_manager: AgentManager):
    agent_ids = [(agent_id, agent_manager.get_info(agent_id).AGENT_NAME) for agent_id in self._ratings['agent_ids']]
    current = self._ratings['current']

    ids_with_avg_rank = (
      (i,
       n,
       sum(current[i].values()) / len(current[i].values()),
       np.median(list(current[i].values())),
       np.percentile(list(current[i].values()), 80))
      for i, n in agent_ids)
    ids_sorted_by_avg_rank = sorted(ids_with_avg_rank, key=lambda x: x[3])

    print('{:>3}: {:<20} {:<5} {:<5} {:<5}'.format(' ', 'agent_id', 'avg', 'med', '80pctl'))
    output = "\n".join(
      '{:>3}: {:<20} {:<5.2f} {:<5.2f} {:<5.2f}'.format(idx + 1, agent_id, agent_name, avg_rank, median_rank, pctl)
      for idx, (agent_id, agent_name, avg_rank, median_rank, pctl) in enumerate(ids_sorted_by_avg_rank))
    print(output)


if __name__ == '__main__':
  agent_manager = AgentManager(file_path='../savefiles/agent_manager.json',
                               models_path=None,
                               possible_agent_names=None)

  lpl = LeaderboardPlacingMatrix(file_path='../savefiles/leaderboards/34.json')
  lpl.load()
  lpl.plot_matrix()
  lpl.print_leaderboard(agent_manager)

  agent_manager.load()

  relevant_leaderboards = ['17', '73', '78', '25', '69', '82', '77']

  for l in relevant_leaderboards:
    # select leaderboard to watch
    leaderboard = LeaderboardTrueskill(file_path='../savefiles/leaderboard_{}.json'.format(l))
    leaderboard.load()

    ratings = [r[0] for r in leaderboard._ratings['current'].values()]
    print('average rating:', sum(ratings) / len(ratings))

    leaderboard.print_current_ratings(agent_manager)

    leaderboard.plot_history(agent_manager)
