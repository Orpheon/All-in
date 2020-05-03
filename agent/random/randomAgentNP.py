import numpy as np

class RandomAgentNP:
  def __init__(self):
    self.rng = np.random.RandomState()

  def act(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser, hole_cards, community_cards):
    self.rng.seed(hole_cards.sum(axis=-1))
    actions = self.rng.randint(0, 3, min_raise.size).astype(int)
    amounts = (self.rng.rand(min_raise.size) * 5).astype(int)

    return actions, amounts

  def start_game(self, batch_size, initial_capital, n_players):
    pass

  def end_trajectory(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser, hole_cards, community_cards, gains):
    pass

  def __str__(self):
    return "RandomAgentNP"