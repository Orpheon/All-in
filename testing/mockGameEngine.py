from league.game import GameEngine


class MockGameEngine(GameEngine):
  def __init__(self, batch_size, initial_capital, small_blind, big_blind, fixed_community_cards, fixed_hole_cards,
               logger):
    super().__init__(batch_size, initial_capital, small_blind, big_blind, logger)
    self._fixed_community_cards = fixed_community_cards
    self._fixed_hole_cards = fixed_hole_cards

  def generate_cards(self):
    return self._fixed_community_cards, self._fixed_hole_cards