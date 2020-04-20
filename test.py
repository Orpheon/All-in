import league.game
import numpy as np
import treys
import time
from agent.sac1.Sac1Agent import Sac1Agent

class Player:
  def __init__(self):
    self.rng = np.random.RandomState()

  def act(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser, hole_cards, community_cards):
    # print("player idx", player_idx)
    # print("hole cards")
    # for hand in hole_cards.tolist():
    #   print("\t"+treys.Card.print_pretty_cards(hand))
    # print("community cards")
    # for hand in community_cards.tolist():
    #   print("\t"+treys.Card.print_pretty_cards(hand))
    self.rng.seed(hole_cards.sum(axis=-1))
    actions = self.rng.randint(0, 3, min_raise.size).astype(int)
    amounts = (self.rng.rand(min_raise.size) * 5).astype(int)

    return actions, amounts

  def start_game(self, batch_size, initial_capital, n_players):
    pass

  def round_end(self, player_idx, round, current_bets, min_raise, prev_round_investment, hole_cards, community_cards):
    pass


batch_size = 10000
game = league.game.GameEngine(BATCH_SIZE=batch_size, INITIAL_CAPITAL=100, SMALL_BLIND=2, BIG_BLIND=4)
t1 = time.time()
scores = game.run_game([Player(), Player(), Player(), Player(), Player(), Player()])
# scores = game.run_game([Sac1Agent(agent_id=1, config={'learning_rate': 0.01}), Player(), Player(), Player(), Player(), Player()])
t2 = time.time()
print("{0} games (computed in {2} seconds):\nMean: {1}\nMedian: {3}".format(batch_size, scores.mean(axis=0), t2 - t1, np.median(scores, axis=0)))