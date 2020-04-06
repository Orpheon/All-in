import league.game
import random

class Player:
  def act(self, player_idx, round, current_bets, min_raise, prev_round_investment, hole_cards, community_cards):
    

game = league.game.GameEngine(BATCH_SIZE=5, INITIAL_CAPITAL=100, SMALL_BLIND=2, BIG_BLIND=4)
game.run_game([0, 0, 0, 0, 0, 0])