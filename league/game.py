import numpy as np
import random

PRE_FLOP = 0
FLOP = 1
TURN = 2
RIVER = 3

FOLD = 0
CALL = 1
RAISE = 2

class GameEngine:
  def __init__(self, BATCH_SIZE, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND):
    self.BATCH_SIZE = BATCH_SIZE
    self.INITIAL_CAPITAL = INITIAL_CAPITAL
    self.SMALL_BLIND = SMALL_BLIND
    self.BIG_BLIND = BIG_BLIND

  def run_game(self, players):
    if len(players) != 6:
      raise ValueError("Only 6 players allowed")
    random.shuffle(players)

    cards = np.tile(np.arange(52), (self.BATCH_SIZE, 1))
    for i in range(self.BATCH_SIZE):
      cards[i, :] = np.random.permutation(cards[i, :])
    community_cards = cards[:, :5]
    hole_cards = np.reshape(cards[:, 5:5 + 2 * len(players)], (self.BATCH_SIZE, len(players), 2))

    winners = np.zeros((self.BATCH_SIZE, len(players)))
    still_playing = np.ones((self.BATCH_SIZE, len(players)))
    total_investment = np.zeros((self.BATCH_SIZE, len(players)))
    visible_community_cards = np.zeros((self.BATCH_SIZE, 5))

    # Pre-flop
    total_investment += self.run_round(players, total_investment, still_playing, PRE_FLOP, hole_cards, visible_community_cards)

    # Flop
    visible_community_cards[:, :3] = community_cards[:, :3]
    total_investment += self.run_round(players, total_investment, still_playing, FLOP, hole_cards, visible_community_cards)

    # Turn
    visible_community_cards[:, 3] = community_cards[:, 3]
    total_investment += self.run_round(players, total_investment, still_playing, TURN, hole_cards, visible_community_cards)

    # River
    total_investment += self.run_round(players, total_investment, still_playing, RIVER, hole_cards, community_cards)

    # Showdown
    pool = np.sum(total_investment, axis=1)
    winner_exists = np.sum(winners, axis=1)
    for game_idx in range(self.BATCH_SIZE):
      if winner_exists[game_idx]:
        winners[
          game_idx, self.evaluate_hands(hole_cards[game_idx, :], community_cards[game_idx], still_playing[game_idx, :])] = 1

    stats = np.subtract(np.multiply(winners, pool), total_investment)
    total_stats = np.sum(stats, axis=1)
    return total_stats

  def run_round(self, players, total_investment, still_playing, round, hole_cards, community_cards):
    current_bets = np.zeros((self.BATCH_SIZE, len(players)))
    max_bets = np.zeros(self.BATCH_SIZE)
    min_raise = np.zeros(self.BATCH_SIZE)
    min_raise[:] = self.BIG_BLIND

    if round == PRE_FLOP:
      current_bets[:, 0] = self.SMALL_BLIND
      current_bets[:, 1] = self.BIG_BLIND
      max_bets[:] = self.BIG_BLIND

    round_countdown = np.zeros(self.BATCH_SIZE)
    round_countdown[:] = len(players)

    while True:
      running_games = np.nonzero(round_countdown > 0)
      for player_idx, player in enumerate(players):
        actions, amounts = player.act(player_idx, round, current_bets, min_raise, total_investment, hole_cards,
                                      community_cards)
        actions = np.multiply(np.multiply(actions, still_playing), running_games)
        illegal_calls = np.nonzero(max_bets + total_investment[:, player_idx] > self.INITIAL_CAPITAL)# And IS_CALL
        actions[illegal_calls] = FOLD

        still_playing[actions == FOLD] = 0

        # TODO Raise
        round_countdown[running_games] -= 1

    return current_bets


  def evaluate_hands(self, hole_cards, community_cards, contenders):
    # TODO get pypokerengine
    return 0
