from agent.baseAgentLoadable import BaseAgentLoadable

import numpy as np

class CallAgentNP(BaseAgentLoadable):

  def __str__(self):
    return 'CallAgent_{0}'.format(self.agent_id)

  @classmethod
  def _config_file_path(cls):
    return './agent/call/config.json'

  def act(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser, hole_cards, community_cards):
    batch_size = current_bets.shape[0]
    actions = np.array([0]*batch_size)
    amounts = np.array([0]*batch_size)
    return actions, amounts
