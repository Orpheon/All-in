import json
from operator import itemgetter
import random
import numpy as np


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
    raise NotImplementedError('League load not implemented')

  def save(self):
    raise NotImplementedError('League save not implemented')

  def clone_mutables(self):
    raise NotImplementedError('League clone_mutables not implemented')


class NormalDivision(Division):

  def __init__(self, file_path, game_engine, leaderboard, agent_manager):
    super().__init__(file_path, game_engine, leaderboard, agent_manager)
    self.agents = {'teachers': [], 'students': []}

  def _generate_matchup(self):
    student = random.choice(self.agents['students'])
    teachers = random.choices(self.agents['teachers'], k=5)
    matchup_ids = [student, *teachers]
    random.shuffle(matchup_ids)
    return [self.agent_manager.get_instance(agent_id) for agent_id in matchup_ids], matchup_ids

  def load(self):
    with open(self.file_path, 'r') as f:
      data = json.load(f)
    self.agents = data
    print('[League <- {}]: loaded T:{} S:{} '.format(self.file_path, len(self.agents['teachers']),
                                                     len(self.agents['students'])))

  def save(self):
    with open(self.file_path, 'w') as f:
      json.dump(self.agents, f, sort_keys=True, indent=2)
    print('[League -> {}]: saved T:{} S:{}'.format(self.file_path, len(self.agents['teachers']),
                                                   len(self.agents['students'])))

  def clone_mutables(self):
    for agent_id in self.agents['students']:
      new_id = self.agent_manager.clone(agent_id)
      self.agents['teachers'].append(new_id)
