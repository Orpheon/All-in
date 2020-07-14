import json
from operator import itemgetter
import random
import numpy as np

from league.agentManager import AgentManager


class Division:

  def __init__(self, file_path, game_engine, leaderboard, agent_manager, division_id):
    self.FILE_PATH = file_path
    self.GAME_ENGINE = game_engine
    self.LEADERBOARD = leaderboard
    self.AGENT_MANAGER = agent_manager
    self.DIVISION_ID = division_id

  def run_next(self):
    matchup, ids = self._generate_matchup()
    placings = self._run_games(matchup, ids)
    self.LEADERBOARD.update_from_placings(placings)

  def _run_games(self, agent_types, agent_ids):
    total_winnings = self.GAME_ENGINE.run_game(agent_types)
    winnings = np.sum(total_winnings, axis=0) / total_winnings.shape[0]

    wins_dict = {aid: (m, []) for aid, m in zip(agent_ids, agent_types)}
    for aid, w in zip(agent_ids, winnings):
      wins_dict[aid][1].append(w)
    avg_wins = [(aid, at, sum(win) / len(win), len(win)) for aid, (at, win) in wins_dict.items()]
    sorted_winnings = sorted(avg_wins, key=lambda x: x[2], reverse=True)

    text_ranking = '\n'.join('{} x {} {:<25}: {:>7.2f}'.format(nw, aid, str(at), avgw)
                             for aid, at, avgw, nw in sorted_winnings)
    print('Winnings: \n{}'.format(text_ranking))

    id_ranking = [(aid, avg_win) for aid, _, avg_win, _ in sorted_winnings]
    return id_ranking

  def _generate_matchup(self):
    raise NotImplementedError('League _generate_matchup not implemented')

  def load(self):
    with open(self.FILE_PATH, 'r') as f:
      data = json.load(f)
    self.state = data
    print('[Division <- {}]: load {}'
          .format(self.FILE_PATH, ' '.join('| {}: {}'.format(key, len(val) if key != 'type' else val)
                                           for key, val in self.state.items())))

  def save(self):
    with open(self.FILE_PATH, 'w') as f:
      json.dump(self.state, f, sort_keys=True, indent=2)
    print('[Division -> {}]: save {}'
          .format(self.FILE_PATH, ' '.join('| {}: {}'.format(key, len(val) if key != 'type' else val)
                                           for key, val in self.state.items())))

  def clone_mutables(self):
    pass


class RandomDivision(Division):

  def __init__(self, file_path, game_engine, leaderboard, agent_manager, division_id):
    super().__init__(file_path, game_engine, leaderboard, agent_manager, division_id)
    self.state = {'type': 'RandomDivision', 'teachers': [], 'students': []}

  def _generate_matchup(self):
    student = random.choice(self.state['students'])
    teachers = random.choices(self.state['teachers'], k=5)
    matchup_ids = [student, *teachers]
    random.shuffle(matchup_ids)
    return [self.AGENT_MANAGER.get_instance(agent_id) for agent_id in matchup_ids], matchup_ids

  def load(self):
    super().load()
    assert self.state['type'] == 'RandomDivision'

  def clone_mutables(self):
    for agent_id in self.state['students']:
      new_id = self.AGENT_MANAGER.clone(agent_id, self.DIVISION_ID)
      self.state['teachers'].append(new_id)


class OverfitDivision(Division):
  def __init__(self, file_path, game_engine, leaderboard, agent_manager: AgentManager, division_id):
    super().__init__(file_path, game_engine, leaderboard, agent_manager, division_id)
    self.state = {'type': 'RandomDivision', 'teacher': None, 'students': [], 'alumni': [], 'is_learning': True}

  def _generate_matchup(self):
    teachers = [self.state['teacher'] for _ in range(5)]

    if self.state['is_learning']:
      student = random.choice(self.state['students'])
    else:
      student = random.choice(self.state['alumni'])

    matchup_ids = [student, *teachers]
    random.shuffle(matchup_ids)
    return [self.AGENT_MANAGER.get_instance(agent_id) for agent_id in matchup_ids], matchup_ids

  def load(self):
    super().load()
    assert self.state['type'] == 'RandomDivision'

  def set_learning(self, teacher):
    self.state['is_learning'] = True
    self.state['teacher'] = teacher

  def set_assessing(self):
    self.state['alumni'] = []
    for s in self.state['students']:
      new_id = self.AGENT_MANAGER.clone(s)
      self.state['alumin'].append(new_id)
    self.state['is_learning'] = False

  def get_best_agent(self):
    return self.LEADERBOARD.get_all_rankings()[0][0]


class PermaEvalChoiceDivision(Division):
  def __init__(self, file_path, game_engine, leaderboard, agent_manager: AgentManager, division_id):
    super().__init__(file_path, game_engine, leaderboard, agent_manager, division_id)
    self.state = {'type': 'PermaEvalChoiceDivision', 'n_rounds_played': {}}

  def _generate_matchup(self):
    all_Teachers = [id for id in self.AGENT_MANAGER.agents if not self.AGENT_MANAGER.get_info(id).TRAINABLE]

    random_agents = random.sample(all_Teachers, k=5)
    agents_sorted_by_usage = sorted([(agent_id, self.state['n_rounds_played'].get(agent_id, 0))
                                     for agent_id in all_Teachers], key=lambda x: x[1])

    least_used_agent = agents_sorted_by_usage[0]

    matchup_ids = [least_used_agent[0], *random_agents]
    random.shuffle(matchup_ids)

    for id in matchup_ids:
      self.state['n_rounds_played'][id] = self.state['n_rounds_played'].get(id, 0) + 1

    return [self.AGENT_MANAGER.get_instance(agent_id) for agent_id in matchup_ids], matchup_ids

  def load(self):
    super().load()
    assert self.state['type'] == 'PermaEvalChoiceDivision'


class PermaEvalSampleDivision(Division):
  def __init__(self, file_path, game_engine, leaderboard, agent_manager: AgentManager, division_id):
    super().__init__(file_path, game_engine, leaderboard, agent_manager, division_id)
    self.state = {'type': 'PermaEvalSampleDivision', 'n_rounds_played': {}}

  def _generate_matchup(self):
    all_Teachers = [id for id in self.AGENT_MANAGER.agents if not self.AGENT_MANAGER.get_info(id).TRAINABLE]

    random_agents = random.sample(all_Teachers, k=5)
    agents_sorted_by_usage = sorted([(agent_id, self.state['n_rounds_played'].get(agent_id, 0))
                                     for agent_id in all_Teachers], key=lambda x: x[1])
    agents_sorted_by_usage_filtered = [(agent_id, nrp) for agent_id, nrp in agents_sorted_by_usage if
                                       agent_id not in random_agents]

    if len(agents_sorted_by_usage_filtered) == 0:
      least_used_agent = agents_sorted_by_usage[0]
    else:
      least_used_agent = agents_sorted_by_usage_filtered[0]

    matchup_ids = [least_used_agent[0], *random_agents]
    random.shuffle(matchup_ids)

    for id in matchup_ids:
      self.state['n_rounds_played'][id] = self.state['n_rounds_played'].get(id, 0) + 1

    return [self.AGENT_MANAGER.get_instance(agent_id) for agent_id in matchup_ids], matchup_ids

  def load(self):
    super().load()
    assert self.state['type'] == 'PermaEvalSampleDivision'


class PermaEvalSimilarDivision(Division):
  RANK_RADIUS = 5

  def __init__(self, file_path, game_engine, leaderboard, agent_manager: AgentManager, division_id):
    super().__init__(file_path, game_engine, leaderboard, agent_manager, division_id)
    self.state = {'type': 'PermaEvalSimilarDivision', 'n_rounds_played': {}}

  def _generate_matchup(self):
    all_Teachers = [id for id in self.AGENT_MANAGER.agents if not self.AGENT_MANAGER.get_info(id).TRAINABLE]
    teachers_by_ts = [(agent_id, self.LEADERBOARD.get_ranking(agent_id)[0]) for agent_id in all_Teachers]
    teachers_sorted_by_ts = list(enumerate(sorted(teachers_by_ts, key=lambda x: x[1])))
    origin = random.choice(teachers_sorted_by_ts)
    origin_agent_id = origin[1][0]
    origin_idx = origin[0]

    top = teachers_sorted_by_ts[origin_idx + 1:origin_idx + 1 + self.RANK_RADIUS]
    bottom = teachers_sorted_by_ts[max(0, origin_idx - self.RANK_RADIUS):origin_idx]

    if len(top) + len(bottom) < 5:
      similar_competition = random.choice(top + bottom, 5)
    else:
      similar_competition = random.sample(top + bottom, 5)
    origin_neighbors = [on[1][0] for on in similar_competition]

    matchup_ids = [origin_agent_id, *origin_neighbors]
    random.shuffle(matchup_ids)

    for id in matchup_ids:
      self.state['n_rounds_played'][id] = self.state['n_rounds_played'].get(id, 0) + 1

    return [self.AGENT_MANAGER.get_instance(agent_id) for agent_id in matchup_ids], matchup_ids

  def load(self):
    super().load()
    assert self.state['type'] == 'PermaEvalSimilarDivision'


class ClimbingDivision(Division):
  P_TRAINING = 0.8
  RANK_RADIUS = 5

  def __init__(self, file_path, game_engine, leaderboard, agent_manager: AgentManager, division_id):
    super().__init__(file_path, game_engine, leaderboard, agent_manager, division_id)
    self.state = {'type': 'ClimbingDivision', 'students': [], 'teachers': []}

  def _generate_matchup(self):
    teachers_by_ts = [(agent_id, self.LEADERBOARD.get_ranking(agent_id)[0]) for agent_id in self.state['teachers']]

    if random.random() <= self.P_TRAINING:
      student = random.choice(self.state['students'])

      sum_ts = sum([ts ** 2 for _, ts in teachers_by_ts])
      p = [ts ** 2 / sum_ts for _, ts in teachers_by_ts]

      teachers = []
      for i in range(5):
        teachers.append(np.random.choice(self.state['teachers'], p=p))
      matchup_ids = [student, *teachers]

    else:
      teachers_sorted_by_ts = list(enumerate(sorted(teachers_by_ts, key=lambda x: x[1])))
      origin = random.choice(teachers_sorted_by_ts)
      origin_agent_id = origin[1][0]
      origin_idx = origin[0]

      top = teachers_sorted_by_ts[origin_idx + 1:origin_idx + 1 + self.RANK_RADIUS]
      bottom = teachers_sorted_by_ts[max(0, origin_idx - self.RANK_RADIUS):origin_idx]

      n_neighbors = 5

      if len(top) + len(bottom) < n_neighbors:
        similar_competition = random.choices(top + bottom, k=n_neighbors)
      else:
        similar_competition = random.sample(top + bottom, k=n_neighbors)
      origin_neighbors = [on[1][0] for on in similar_competition]
      matchup_ids = [origin_agent_id, *origin_neighbors]

    random.shuffle(matchup_ids)
    return [self.AGENT_MANAGER.get_instance(agent_id) for agent_id in matchup_ids], matchup_ids

  def load(self):
    super().load()
    assert self.state['type'] == 'ClimbingDivision'

  def clone_mutables(self):
    for agent_id in self.state['students']:
      new_id = self.AGENT_MANAGER.clone(agent_id, self.DIVISION_ID)
      self.state['teachers'].append(new_id)
