import json
from operator import itemgetter
import random
import numpy as np

from league.agentManager import AgentManager


class Division:

  def __init__(self, file_path, game_engine, leaderboard, agent_manager):
    self.file_path = file_path
    self.game_engine = game_engine
    self.leaderboard = leaderboard
    self.agent_manager = agent_manager

  def run_next(self):
    matchup, ids = self._generate_matchup()
    placings = self._run_games(matchup, ids)
    self.leaderboard.update_from_placings(placings)

  def _run_games(self, agent_types, agent_ids):
    total_winnings = self.game_engine.run_game(agent_types)
    winnings = np.sum(total_winnings, axis=0) / total_winnings.shape[0]

    wins_dict = {aid: (m, []) for aid, m in zip(agent_ids, agent_types)}
    for aid, w in zip(agent_ids, winnings):
      wins_dict[aid][1].append(w)
    avg_wins = [(aid, at, sum(win) / len(win), len(win)) for aid, (at, win) in wins_dict.items()]
    sorted_winnings = sorted(avg_wins, key=lambda x: x[2], reverse=True)

    text_ranking = '\n'.join('{} x {} {:>10}: {:>7.2f}'.format(nw, aid, str(at), avgw)
                             for aid, at, avgw, nw in sorted_winnings)
    print('Winnings: \n{}'.format(text_ranking))

    id_ranking = [aid for aid, _, _, _ in sorted_winnings]
    return id_ranking

  def _generate_matchup(self):
    raise NotImplementedError('League _generate_matchup not implemented')

  def load(self):
    with open(self.file_path, 'r') as f:
      data = json.load(f)
    self.state = data
    print('[League <- {}]: loaded '.format(self.file_path))

  def save(self):
    with open(self.file_path, 'w') as f:
      json.dump(self.state, f, sort_keys=True, indent=2)
    print('[League -> {}]: saved'.format(self.file_path))

  def clone_mutables(self):
    raise NotImplementedError('League clone_mutables not implemented')


class RandomDivision(Division):

  def __init__(self, file_path, game_engine, leaderboard, agent_manager):
    super().__init__(file_path, game_engine, leaderboard, agent_manager)
    self.state = {'type': 'RandomDivision', 'teachers': [], 'students': []}

  def _generate_matchup(self):
    student = random.choice(self.state['students'])
    teachers = random.choices(self.state['teachers'], k=5)
    matchup_ids = [student, *teachers]
    random.shuffle(matchup_ids)
    return [self.agent_manager.get_instance(agent_id) for agent_id in matchup_ids], matchup_ids

  def load(self):
    super().load()
    assert self.state['type'] == 'RandomDivision'

  def clone_mutables(self):
    for agent_id in self.state['students']:
      new_id = self.agent_manager.clone(agent_id)
      self.state['teachers'].append(new_id)


class OverfitDivision(Division):
  def __init__(self, file_path, game_engine, leaderboard, agent_manager: AgentManager):
    super().__init__(file_path, game_engine, leaderboard, agent_manager)
    self.state = {'type': 'RandomDivision', 'teacher': None, 'students': [], 'alumni': [], 'is_learning': True}

  def _generate_matchup(self):
    teachers = [self.state['teacher'] for _ in range(5)]

    if self.state['is_learning']:
      student = random.choice(self.state['students'])
    else:
      student = random.choice(self.state['alumni'])

    matchup_ids = [student, *teachers]
    random.shuffle(matchup_ids)
    return [self.agent_manager.get_instance(agent_id) for agent_id in matchup_ids], matchup_ids

  def load(self):
    super().load()
    assert self.state['type'] == 'RandomDivision'

  def set_learning(self, teacher):
    self.state['is_learning'] = True
    self.state['teacher'] = teacher

  def set_assessing(self):
    self.state['alumni'] = []
    for s in self.state['students']:
      new_id = self.agent_manager.clone(s)
      self.state['alumin'].append(new_id)
    self.state['is_learning'] = False

  def get_best_agent(self):
    return self.leaderboard.get_rankings()[0][0]


class PermaEvalChoiceDivision(Division):
  def __init__(self, file_path, game_engine, leaderboard, agent_manager: AgentManager):
    super().__init__(file_path, game_engine, leaderboard, agent_manager)
    self.state = {'type': 'PermaEvalChoiceDivision', 'n_rounds_played': {}}

  def _generate_matchup(self):
    all_Teachers = [id for id in self.agent_manager.agents if not self.agent_manager.get_info(id).TRAINABLE]

    random_agents = random.sample(all_Teachers, k=5)
    agents_sorted_by_usage = sorted([(agent_id, self.state['n_rounds_played'].get(agent_id, 0))
                                     for agent_id in all_Teachers], key=lambda x: x[1])

    least_used_agent = agents_sorted_by_usage[0]

    matchup_ids = [least_used_agent[0], *random_agents]
    random.shuffle(matchup_ids)

    for id in matchup_ids:
      self.state['n_rounds_played'][id] = self.state['n_rounds_played'].get(id, 0) + 1

    return [self.agent_manager.get_instance(agent_id) for agent_id in matchup_ids], matchup_ids

  def load(self):
    super().load()
    assert self.state['type'] == 'PermaEvalChoiceDivision'


class PermaEvalSampleDivision(Division):
  def __init__(self, file_path, game_engine, leaderboard, agent_manager: AgentManager):
    super().__init__(file_path, game_engine, leaderboard, agent_manager)
    self.state = {'type': 'PermaEvalSampleDivision', 'n_rounds_played': {}}

  def _generate_matchup(self):
    all_Teachers = [id for id in self.agent_manager.agents if not self.agent_manager.get_info(id).TRAINABLE]

    random_agents = random.sample(all_Teachers, k=5)
    agents_sorted_by_usage = sorted([(agent_id, self.state['n_rounds_played'].get(agent_id, 0))
                                     for agent_id in all_Teachers], key=lambda x: x[1])
    agents_sorted_by_usage_filtered = [(agent_id, nrp) for agent_id, nrp in agents_sorted_by_usage if agent_id not in random_agents]

    if len(agents_sorted_by_usage_filtered) == 0:
      least_used_agent = agents_sorted_by_usage[0]
    else:
      least_used_agent = agents_sorted_by_usage_filtered[0]

    matchup_ids = [least_used_agent[0], *random_agents]
    random.shuffle(matchup_ids)

    for id in matchup_ids:
      self.state['n_rounds_played'][id] = self.state['n_rounds_played'].get(id, 0) + 1

    return [self.agent_manager.get_instance(agent_id) for agent_id in matchup_ids], matchup_ids

  def load(self):
    super().load()
    assert self.state['type'] == 'PermaEvalSampleDivision'
