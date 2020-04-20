import json

import numpy as np


class Logger:
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

  def add_action(self, round, player_idx, actions, amounts, round_countdown):
    print({'round': round, 'player_idx': player_idx, 'actions': np.copy(actions),
           'amounts': np.copy(amounts), 'round_countdown': np.copy(round_countdown)})
    self.action_history.append({'round': round, 'player_idx': player_idx, 'actions': np.copy(actions),
                                'amounts': np.copy(amounts), 'round_countdown': np.copy(round_countdown)})

  def append_folded(self, after_round, folded):
    self.folded.append((after_round, np.copy(folded)))

  def save_to_file(self):
    with open(self.path, 'w') as f:
      f.write(
        json.dumps({'hole_cards': self.hole_cards, 'community_cards': self.community_cards,
                    'action_history': self.action_history})
      )
