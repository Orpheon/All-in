import numpy as np
import random
import pickle
import treys

PRE_FLOP = 0
FLOP = 1
TURN = 2
RIVER = 3

FOLD = 0
CALL = 1
RAISE = 2
ALLIN = 3

FULL_DECK = np.array(treys.Deck.GetFullDeck())

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
      cards[i, :] = FULL_DECK[np.random.permutation(cards[i, :])]
    community_cards = cards[:, :5]
    hole_cards = np.reshape(cards[:, 5:5 + 2 * len(players)], (self.BATCH_SIZE, len(players), 2))

    with open("tmp_cards.dump", "wb") as f:
      pickle.dump((community_cards, hole_cards), f)

    with open("tmp_cards.dump", "rb") as f:
      community_cards, hole_cards = pickle.load(f)

    winners = np.zeros((self.BATCH_SIZE, len(players)))
    still_playing = np.ones((self.BATCH_SIZE, len(players)))
    prev_round_investment = np.zeros((self.BATCH_SIZE, len(players)))

    print("PREFLOP")

    # Pre-flop
    prev_round_investment += self.run_round(players, prev_round_investment, still_playing, PRE_FLOP, hole_cards, community_cards[:, :0])

    print("FLOP")

    # Flop
    prev_round_investment += self.run_round(players, prev_round_investment, still_playing, FLOP, hole_cards, community_cards[:, :3])

    print("TURN")

    # Turn
    prev_round_investment += self.run_round(players, prev_round_investment, still_playing, TURN, hole_cards, community_cards[:, :4])

    print("RIVER")

    # River
    prev_round_investment += self.run_round(players, prev_round_investment, still_playing, RIVER, hole_cards, community_cards)

    print("SHOWDOWN")

    # Showdown
    pool = np.sum(prev_round_investment, axis=1)
    print("Total pool", pool)
    # TODO: WINNING
    # winner_exists = np.sum(winners, axis=1)
    # for game_idx in range(self.BATCH_SIZE):
    #   if winner_exists[game_idx]:
    #     winners[
    #       game_idx, self.evaluate_hands(hole_cards[game_idx, :], community_cards[game_idx], still_playing[game_idx, :])] = 1
    #
    # stats = np.subtract(np.multiply(winners, pool), prev_round_investment)
    # total_stats = np.sum(stats, axis=1)
    # return total_stats
    return pool

  def run_round(self, players, prev_round_investment, still_playing, round, hole_cards, community_cards):
    current_bets = np.zeros((self.BATCH_SIZE, len(players)))
    max_bets = np.zeros(self.BATCH_SIZE)
    min_raise = np.zeros(self.BATCH_SIZE)
    min_raise[:] = self.BIG_BLIND
    allin_poolsize = np.zeros((self.BATCH_SIZE, len(players)))
    current_allin_check = np.zeros((self.BATCH_SIZE, len(players)))

    if round == PRE_FLOP:
      current_bets[:, 0] = self.SMALL_BLIND
      current_bets[:, 1] = self.BIG_BLIND
      max_bets[:] = self.BIG_BLIND

    round_countdown = np.zeros(self.BATCH_SIZE)
    round_countdown[:] = len(players)

    while True:
      running_games = np.nonzero(round_countdown > 0)[0]
      for player_idx, player in enumerate(players):
        actions, amounts = player.act(player_idx, round, current_bets, min_raise, prev_round_investment,
                                      hole_cards[:, player_idx, :], community_cards)

        round_countdown[running_games] -= 1

        actions *= (round_countdown > 0) * still_playing[:, player_idx]

        print("Player", player_idx)
        print("Actions", actions)

        ###########
        # RAISING #
        ###########

        # If player wants to raise, first set the action of all those that can't afford it to all-in, then raise the remainder and reset round counter
        false_raises = np.where(
          np.logical_and(
            actions == RAISE,
            max_bets + min_raise > self.INITIAL_CAPITAL - prev_round_investment[:, player_idx]
          )
        )[0]
        if false_raises.size > 0:
          print("False raises", false_raises)
          # All of these players can't afford to raise properly, so they do a false raise and all-in
          actions[false_raises] = ALLIN
          investment = self.INITIAL_CAPITAL - prev_round_investment[false_raises, player_idx]
          max_bets[false_raises] = np.maximum(investment, max_bets[false_raises])
          current_allin_check[false_raises, player_idx] = 1
          current_bets[false_raises, player_idx] = investment

        # Handle the real raises
        raises = np.where(actions == RAISE)[0]
        if raises.size > 0:
          print("True raises", raises, amounts[raises])
          investment = np.maximum(current_bets[raises, player_idx] + amounts[raises], max_bets[raises] + min_raise[raises])
          # Isolate all the games where someone had just gone allin and save for them the sidepool
          sidepool_indices = np.nonzero(current_allin_check[raises, :])
          allin_poolsize[sidepool_indices] = (np.sum(prev_round_investment, axis=1) + np.sum(current_bets, axis=1))[sidepool_indices[0]]
          current_allin_check[raises, :] = 0
          # Reset the bets and countdown
          max_bets[raises] = investment
          current_bets[raises, player_idx] = investment
          round_countdown[raises] = len(players)

        ###########
        # CALLING #
        ###########

        # If player wants to call/check, first set the action of all those that can't afford it to allin, then check, then look if round end condition is met
        false_calls = np.where(
          np.logical_and(
            actions == CALL,
            prev_round_investment[:, player_idx] + (max_bets - current_bets[:, player_idx]) > self.INITIAL_CAPITAL
          )
        )[0]
        if false_calls.size > 0:
          print("False calls", false_calls)
          # All of these players can't actually afford to call, so they go all-in instead
          actions[false_calls] = FOLD
          investment = self.INITIAL_CAPITAL - prev_round_investment[false_calls, player_idx]
          current_allin_check[false_calls, player_idx] = 1
          current_bets[false_calls, player_idx] = investment

        # Handle the real checks
        calls = np.where(actions == CALL)[0]
        if calls.size > 0:
          print("True calls", calls)
          investment = max_bets[calls]
          # Reset the bets and countdown
          max_bets[calls] = investment
          current_bets[calls, player_idx] = investment

        ###########
        # FOLDING #
        ###########

        still_playing[np.where(actions == FOLD)[0], player_idx] = 0
        still_playing[np.where(actions == ALLIN)[0], player_idx] = 0

        print("Bets after turn", current_bets[:, player_idx])

        if np.sum(round_countdown[running_games]) <= 0:
          return current_bets


  def evaluate_hands(self, hole_cards, community_cards, contenders):
    # TODO get pypokerengine
    return 0
