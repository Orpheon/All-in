import numpy as np

from agent.baseAgentNP import BaseAgentNP

RAISE_ACTION = 2
MAX_RAISE = 200


class AllinAgentNP(BaseAgentNP):

  def __str__(self):
    return 'AllIn {}'.format('T' if self.trainable else 'N')

  def act(self, player_idx, round, active_rounds, current_bets, min_raise, prev_round_investment, folded, last_raiser,
          hole_cards, community_cards):
    actions = np.full_like(min_raise, RAISE_ACTION)
    amounts = np.full_like(min_raise, MAX_RAISE)
    return actions, amounts
