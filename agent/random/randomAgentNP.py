import numpy as np

from agent.baseAgentNP import BaseAgentNP


class RandomAgentNP(BaseAgentNP):
  rng = np.random.RandomState()

  def __str__(self):
    return 'Rndm {} {}'.format('T' if self.trainable else 'N', super().__str__())

  def act(self, player_idx, round, active_rounds, current_bets, min_raise, prev_round_investment, folded, last_raiser,
          hole_cards, community_cards):
    actions = self.rng.randint(0, 3, min_raise.size).astype(int)

    max_raise = 200 - current_bets[:, player_idx] + prev_round_investment[:, player_idx]

    amounts = (self.rng.rand(min_raise.size) * (max_raise - min_raise) + min_raise).astype(int)
    return actions, amounts
