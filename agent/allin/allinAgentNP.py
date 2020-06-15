import numpy as np

from agent.baseAgentLoadable import BaseAgentLoadable

RAISE_ACTION = 2
MAX_RAISE = 200


class AllinAgentNP(BaseAgentLoadable):

  def act(self, player_idx, round, active_rounds, current_bets, min_raise, prev_round_investment, folded, last_raiser,
          hole_cards, community_cards):
    actions = np.full_like(min_raise, RAISE_ACTION)
    amounts = np.full_like(min_raise, MAX_RAISE)
    return actions, amounts

  def __str__(self):
    return 'AllinAgentNP_{0}'.format(self.agent_id)

  @classmethod
  def _config_file_path(cls):
    return './agent/allin/config.json'
