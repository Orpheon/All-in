from typing import List, Dict, Union, Tuple
from agent.baseAgentNP import BaseAgentNP

class BaseAgentActionLogger(BaseAgentNP):

  def __init__(self, agentNP):
    super().__init__()
    self._agentNP: BaseAgentNP = agentNP

  def act(self, player_idx, round, active_rounds, current_bets, min_raise, prev_round_investment, folded, last_raiser,
          hole_cards, community_cards):
    action, amount = self._agentNP.act(player_idx, round, active_rounds, current_bets, min_raise, prev_round_investment,
                                       folded, last_raiser, hole_cards, community_cards)
    return action, amount

  def initialize(self, batch_size, initial_capital, n_players):
    pass

  def end_trajectory(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser, hole_cards, community_cards, gains):
    pass

  def spawn_clone(self):
    pass
