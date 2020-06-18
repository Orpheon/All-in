import numpy as np

from agent.baseAgentLoadable import BaseAgentLoadable


class RandomAgentNP(BaseAgentLoadable):

  def __init__(self, agent_id, config):
    super().__init__(agent_id, config)

    self.rng = np.random.RandomState()

  def act(self, player_idx, round, active_rounds, current_bets, min_raise, prev_round_investment, folded, last_raiser,
          hole_cards, community_cards):
    actions = self.rng.randint(0, 3, min_raise.size).astype(int)

    max_raise = 200 - current_bets[:, player_idx] + prev_round_investment[:, player_idx]

    amounts = (self.rng.rand(min_raise.size) * (max_raise-min_raise) + min_raise).astype(int)
    return actions, amounts

  @classmethod
  def _config_file_path(cls):
    return './agent/random/config.json'
