import numpy as np

from agent.baseAgentNP import BaseAgentNP


class FoldAgentNP(BaseAgentNP):

  def __init__(self):
    super().__init__()
    self.trainable = False
    self.model_path = ''

  @classmethod
  def get_instance(cls, trainable, model_path):
    return cls()

  def __str__(self):
    return 'Qlearn1 {}'.format('T' if self.trainable else 'N')

  def act(self, player_idx, round, active_games, current_bets, min_raise, prev_round_investment, folded, last_raiser,
          hole_cards, community_cards):
    actions = np.zeros_like(min_raise)
    amounts = np.zeros_like(min_raise)
    return actions, amounts
