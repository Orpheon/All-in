import numpy as np

from agent.baseAgentLoadable import BaseAgentLoadable


class RandomAgentNP(BaseAgentLoadable):

  def __init__(self, agent_id, config):
    super().__init__(agent_id, config)

    self.rng = np.random.RandomState()

  def act(self, player_idx, round, active_rounds, current_bets, min_raise, prev_round_investment, folded, last_raiser,
          hole_cards, community_cards):
    actions = self.rng.randint(0, 3, min_raise.size).astype(int)
    amounts = (self.rng.rand(min_raise.size) * 5).astype(int)
    return actions, amounts

  def __str__(self):
    return 'RandomAgentNP_{0}'.format(self.agent_id)

  @classmethod
  def _config_file_path(cls):
    return './agent/random/config.json'
