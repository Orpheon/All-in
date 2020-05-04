import numpy as np

from agent.baseAgentLoadable import BaseAgentLoadable


class CallAgentNP(BaseAgentLoadable):

  def act(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser, hole_cards, community_cards):
    actions = np.ones_like(min_raise)
    amounts = np.zeros_like(min_raise)
    return actions, amounts

  def __str__(self):
    return 'CallAgentNP_{0}'.format(self.agent_id)

  @classmethod
  def _config_file_path(cls):
    return './agent/call/config.json'
