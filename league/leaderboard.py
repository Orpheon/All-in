import json
from trueskill import TrueSkill
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from league.agentManager import AGENT_TYPES, AgentManager

TRUESKILL_START_MU = 100
TRUESKILL_START_SIGMA = 30


class Leaderboard:

  def __init__(self, file_path):
    self.FILE_PATH = file_path
    self._trueskill = TrueSkill(mu=TRUESKILL_START_MU, sigma=TRUESKILL_START_SIGMA)
    self._ratings = {'current': {}, 'history': []}

  def load(self):
    with open(self.FILE_PATH, 'r') as f:
      data = json.load(f)
    self._ratings = data
    print('[Leaderboard <- {}]: loaded {} rankings over {} games'.format(self.FILE_PATH, len(self._ratings['current']),
                                                                         len(self._ratings['history'])))

  def save(self):
    with open(self.FILE_PATH, 'w') as f:
      json.dump(self._ratings, f)
    print('[Leaderboard -> {}]: saved {} rankings over {} games'.format(self.FILE_PATH, len(self._ratings['current']),
                                                                        len(self._ratings['history'])))

  def update_from_placings(self, agents):
    current_ratings = [self._trueskill.Rating(*self._ratings['current'].get(agent, ())) for agent in agents]
    new_rating_groups = self._trueskill.rate([(cr,) for cr in current_ratings])
    hist_append = {}
    for agent, new_rating_group in zip(agents, new_rating_groups):
      new_rating = (new_rating_group[0].mu, new_rating_group[0].sigma)
      self._ratings['current'][agent] = new_rating
      hist_append[agent] = new_rating
    self._ratings['history'].append(hist_append)

  def get_rankings(self):
    sorted_ratings = sorted(self._ratings['current'].items(), key=lambda x: x[1][0], reverse=True)
    return sorted_ratings

  def reset_rankings(self):
    self._ratings = {'current': {}, 'history': []}

  def plot_history(self, agent_manager):

    plt.figure(0)
    plt.subplot()

    mu = np.ndarray((len(self._ratings['history']),))
    sigma_2 = np.ndarray((len(self._ratings['history']),))
    x = np.arange(len(self._ratings['history']))

    agent_types = {agent_type: idx for idx, agent_type in enumerate(AGENT_TYPES.keys())}
    palette = sns.color_palette("husl", n_colors=len(agent_types))
    # palette = [(60,180,75), (255,225,25), (0,130,200), (245,130,48), (220,190,255), (128,0,0), (0,0,128), (128,128,128), (0,0,0)]
    # palette = [(r/255, g/255, b/255) for r,g,b in palette]
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
    ratings = [(v[0], v[1], k, agent_manager.get_info(k).AGENT_TYPE, 'S' if agent_manager.get_info(k).TRAINABLE else 'T') for k, v in self._ratings['current'].items()]
    output = "\n".join("{:>3}: {:>10} {} {}: {:>7.2f} +/- {:>5.2f}".format(idx+1, at, aid, tr, rating, 2*sd)
                       for idx, (rating, sd, aid, at, tr) in enumerate(sorted(ratings, reverse=True)))
    print(output)


if __name__ == '__main__':
  agent_manager = AgentManager('../savefiles/agent_manager.json', None)
  agent_manager.load()
  leaderboard = Leaderboard('../savefiles/leaderboard_1.json')
  leaderboard.load()

  ratings = [r[0] for r in leaderboard._ratings['current'].values()]
  print(sum(ratings)/len(ratings))

  leaderboard.print_current_ratings(agent_manager)

  leaderboard.plot_history(agent_manager)
