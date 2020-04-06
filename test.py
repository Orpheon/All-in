import league.game
import numpy as np

class Player:
  def __init__(self):
    self.rng = np.random.RandomState()

  def act(self, player_idx, round, current_bets, min_raise, prev_round_investment, hole_cards, community_cards):
    # print("player idx", player_idx)
    # print("hole cards", hole_cards)
    # print("community cards", community_cards)
    self.rng.seed(hole_cards.sum(axis=-1))
    actions = self.rng.randint(0, 3, min_raise.size).astype("float")
    amounts = self.rng.rand(min_raise.size) * 5

    return actions, amounts


game = league.game.GameEngine(BATCH_SIZE=5, INITIAL_CAPITAL=100, SMALL_BLIND=2, BIG_BLIND=4)
game.run_game([Player(), Player(), Player(), Player(), Player(), Player()])