import numpy as np

from agent.baseAgentNP import BaseAgentNP


class FoldAgentNP(BaseAgentNP):

  def act(self, player_idx, round, active_games, current_bets, min_raise, prev_round_investment, folded, last_raiser,
          hole_cards, community_cards):
    actions = np.zeros_like(min_raise)
    amounts = np.zeros_like(min_raise)
    return actions, amounts
