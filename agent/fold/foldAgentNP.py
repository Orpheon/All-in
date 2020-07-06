import numpy as np

from agent.baseAgentNP import BaseAgentNP


class FoldAgentNP(BaseAgentNP):

  def __str__(self):
    return 'Fold {} {}'.format('T' if self.trainable else 'N', super().__str__())

  def act(self, player_idx, round, active_rounds, current_bets, min_raise, prev_round_investment, folded, last_raiser,
          hole_cards, community_cards):
    actions = np.zeros_like(min_raise)
    amounts = np.zeros_like(min_raise)
    return actions, amounts
