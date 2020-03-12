import json
import random

import matplotlib.pyplot as plt
import trueskill


class Rating:

  def __init__(self, file_path):
    self._file_path = file_path
    self._ratings = {'all_agents': set(), 'history': []}
    self._trueskill = trueskill.TrueSkill(mu=100, sigma=30, )

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

  def plot_history(self):
    datapoints = len(self._ratings['history'])
    plt.figure(0)
    plt.subplot()

    for agent in self._ratings['all_agents']:
      y_ratings = [e.get(agent, None) for e in self._ratings['history'] if agent in e]
      x_numbers = range(datapoints-len(y_ratings)+1, datapoints+1)
      plt.errorbar(x=x_numbers,
                   y=[r['mu'] for r in y_ratings],
                   yerr=[r['sigma'] for r in y_ratings])
    plt.legend(self._ratings['all_agents'])
    plt.show()

  def update_from_placings(self, agents_placing):
    agents_ratings = {}
    for agent in agents_placing:
      if agent in self._ratings['all_agents'] and len(self._ratings['history']) != 0:
        agent_rating = self._ratings['history'][-1][agent]
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
    if len(self._ratings['history']) != 0:
      update_map = {k: v for k, v in self._ratings['history'][-1].items()}
    else:
      update_map = {}

    for agent_id, rating in update_values.items():
      self._ratings['all_agents'].add(agent_id)
      update_map[agent_id] = {'mu': rating.mu, 'sigma': rating.sigma}
    self._ratings['history'].append(update_map)


if __name__ == '__main__':
  rating = Rating('ratings.json')
  players = ['player{0}'.format(n) for n in range(2, 7)]

  for _ in range(10):
    random.shuffle(players)
    rating.update_from_placings(['player1', *players])

  rating.plot_history()
