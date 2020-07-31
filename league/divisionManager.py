import json
from collections import namedtuple
import random

import numpy as np

from league.division import RandomDivision
from league.division import OverfitDivision
from league.division import PermaEvalSimilarDivision
from league.division import PermaEvalSampleDivision
from league.division import ClimbingDivision

from league.leaderboard import LeaderboardTrueskill, LeaderboardWinningsMatrix
from league.leaderboard import LeaderboardPlacingMatrix

DIVISION_TYPES = {'Random': RandomDivision,
                  'Overfit': OverfitDivision,
                  'Climbing': ClimbingDivision,
                  'PESimilar': PermaEvalSimilarDivision,
                  'PESample': PermaEvalSampleDivision}

LEADERBOARD_TYPES = {'Trueskill': LeaderboardTrueskill,
                     'PlacingMatrix': LeaderboardPlacingMatrix,
                     'WinningsMatrix': LeaderboardWinningsMatrix}

DivisionInfo = namedtuple('DivisionInfo', ['DIVISION_TYPE', 'FILE_PATH', 'LEADERBOARDS'])


class DivisionManager:

  def __init__(self, file_path, divis_path, leaderboards_path):
    self.FILE_PATH = file_path
    self.DIVIS_PATH = divis_path
    self.LEADERBOARDS_PATH = leaderboards_path
    self.divis = {}

  def add_division(self, division_type, leaderboard_types):
    all_possible_divi_ids = {'{:02}'.format(i) for i in range(100)}  # TODO currently limited to 100 divisions
    taken = self.divis.keys()
    available = list(all_possible_divi_ids - taken)
    divi_id = random.choice(available)
    self.divis[divi_id] = DivisionInfo(division_type,
                                       '{}/{}.json'.format(self.DIVIS_PATH, divi_id),
                                       [(lb_type, '{}/{}_{}.json'.format(self.LEADERBOARDS_PATH, divi_id, lb_idx))
                                        for lb_idx, lb_type in enumerate(leaderboard_types)])
    return divi_id

  def print_available_divisions(self):
    print('[DivisionManager] available divisions: {}'.format(len(self.divis)))
    for k, v in self.divis.items():
      print(k, v)

  def _instantiate_division(self, division_info: DivisionInfo, game_engine, agent_manager, divi_id):
    leaderboards = [LEADERBOARD_TYPES[lb_type](lb_type, lb_path)
                    for lb_type, lb_path in division_info.LEADERBOARDS]
    divi = DIVISION_TYPES[division_info.DIVISION_TYPE](division_info.FILE_PATH, game_engine, leaderboards, agent_manager,
                                                       divi_id)
    return divi, leaderboards

  def get_divi_instances(self, divi_ids, game_engine, agent_manager):
    divis = {divi_id: (self._instantiate_division(self.divis[divi_id], game_engine, agent_manager, divi_id))
             for divi_id in divi_ids}
    return divis

  def load(self):
    with open(self.FILE_PATH, 'r') as f:
      data = json.load(f)
    self.divis = {k: DivisionInfo(*v) for k, v in data.items()}
    print('[DivisionManager <- {}]: loaded {} divis'.format(self.FILE_PATH, len(self.divis)))

  def save(self):
    with open(self.FILE_PATH, 'w') as f:
      json.dump(self.divis, f, sort_keys=True, indent=2)
    print('[DivisionManager -> {}]: saved {} divis'.format(self.FILE_PATH, len(self.divis)))
