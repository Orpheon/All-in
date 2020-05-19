import json
import random
import numpy as np

import matplotlib.pyplot as plt
import trueskill
import seaborn as sns

sns.set_style("whitegrid")

TRUESKILL_START_MU = 100
TRUESKILL_START_SIGMA = 30


class Rating:

  def __init__(self, file_path):
    self._file_path = file_path
    self._ratings = {'all_agents': set(), 'history': [], 'latest': {}}
    self._trueskill = trueskill.TrueSkill(mu=TRUESKILL_START_MU, sigma=TRUESKILL_START_SIGMA, )

    # TODO: configure trueskill environment parameters

  def load_ratings(self):
    with open(self._file_path, 'r') as f:
      tmp = json.loads(f.read())
      tmp['all_agents'] = set(tmp['all_agents'])
      self._ratings = dict(tmp)

  def save_ratings(self):
    with open(self._file_path, 'w') as f:
      tmp = dict(self._ratings)
      tmp['all_agents'] = list(tmp['all_agents'])
      f.write(json.dumps(tmp))

  def get_rating_from_id(self, id):
    return self._ratings['latest'].get(id, {'mu': TRUESKILL_START_MU, 'sigma': TRUESKILL_START_SIGMA})

  def plot_history(self):
    plt.figure(0)
    plt.subplot()

    mu = np.ndarray((len(self._ratings['history']),))
    sigma_2 = np.ndarray((len(self._ratings['history']),))
    x = np.arange(len(self._ratings['history']))

    for agent in self._ratings['all_agents']:
      for idx,game in enumerate(self._ratings['history']):
        if agent in game:
          mu[idx] = game[agent]['mu']
          sigma_2[idx] = game[agent]['sigma'] * 2
        else:
          mu[idx] = np.NaN
          sigma_2[idx] = np.NaN
      mask = np.isfinite(mu)
      sns.lineplot(x=x[mask], y=mu[mask])
      # plt.errorbar(x=x[mask],
      #              y=mu[mask],
      #              yerr=sigma_2[mask],
      #              linestyle='-')
      plt.ylim(87, 125)
    plt.legend(self._ratings['all_agents'])
    plt.show()

  def update_from_placings(self, agents_placing):
    agents_ratings = {}
    for agent in agents_placing:
      if agent in self._ratings['all_agents'] and len(self._ratings['history']) != 0:
        for hist in reversed(self._ratings['history']):
          if agent in hist:
            agent_rating = hist[agent]
            break
        agents_ratings[agent] = self._trueskill.Rating(mu=agent_rating['mu'], sigma=agent_rating['sigma'])
      else:
        agents_ratings[agent] = self._trueskill.Rating()
    rating_groups = [(agents_ratings[agent],) for agent in agents_placing]
    new_ratings = self._trueskill.rate(rating_groups, ranks=range(len(agents_placing)))
    self._update_ranks({agent: new_ratings[i][0] for i, agent in enumerate(agents_placing)})

  def _update_ranks(self, update_values):
    '''
    :param update_values: [{'player1': trueskill.Rating}]
    '''
    update_map = {}

    for agent_id, rating in update_values.items():
      self._ratings['all_agents'].add(agent_id)
      self._ratings['latest'][agent_id] = {'mu': rating.mu, 'sigma': rating.sigma}
      update_map[agent_id] = {'mu': rating.mu, 'sigma': rating.sigma}
    self._ratings['history'].append(update_map)


if __name__ == '__main__':
  rating = Rating('runner_ratings.json')
  rating.load_ratings()
  rating.plot_history()
