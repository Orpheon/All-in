import json

import numpy as np
import datetime
import os
import pickle
import bz2


class GenericLogger:
  def __init__(self, root='./gamelogs'):
    self._root = root
    self._log = []
    os.makedirs(self._root, exist_ok=True)

  def log(self, event_code, data):
    self._log.append((event_code, *data))

  def save_to_file(self):
    today_str = datetime.datetime.now().isoformat()
    today_str = today_str.replace(':', '-')  # windows compatibility
    file_path = os.path.join(self._root, today_str + ".bz2")
    with bz2.BZ2File(file_path, 'wb') as f:
      pickle.dump(self._log, f)
    self._log.clear()


class NoneLogger:
  def __init__(self):
    pass

  def log(self, event_code, data):
    pass

  def save_to_file(self):
    pass

class OldLogger:
  hole_cards = None
  community_cards = None
  action_history = []
  folded = None

  def __init__(self, path):
    self.path = path

  def start_new_game(self, players, batch_size):
    self.hole_cards = None
    self.community_cards = None
    self.action_history = []
    self.folded = []

  def set_cards(self, community_cards, hole_cards):
    if self.community_cards is not None: raise PermissionError()
    if self.hole_cards is not None: raise PermissionError()
    self.community_cards = np.copy(community_cards)
    self.hole_cards = np.copy(hole_cards)

  def add_action(self, round, player_idx, actions, amounts, round_countdown, folded):
    print({'round': round, 'player_idx': player_idx, 'actions': np.copy(actions),
           'amounts': np.copy(amounts), 'round_countdown': np.copy(round_countdown), 'folded': folded})
    self.action_history.append({'round': round, 'player_idx': player_idx, 'actions': np.copy(actions),
                                'amounts': np.copy(amounts), 'round_countdown': np.copy(round_countdown),
                                'folded': np.copy(folded)})

  def append_folded(self, after_round, folded):
    print(after_round, np.copy(folded))
    self.folded.append((after_round, np.copy(folded)))

  def save_to_file(self):
    with open(self.path, 'w') as f:
      f.write(
        json.dumps({'hole_cards': self.hole_cards, 'community_cards': self.community_cards,
                    'action_history': self.action_history})
      )
