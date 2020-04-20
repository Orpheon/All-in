import numpy as np
import random
import pickle
import treys

# FIXME Remove this and add constants.* to everything once merge conflicts no longer an issue
from constants import *

FULL_DECK = np.array(treys.Deck.GetFullDeck())

class GameEngine:
  def __init__(self, BATCH_SIZE, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND):
    self.BATCH_SIZE = BATCH_SIZE
    self.INITIAL_CAPITAL = INITIAL_CAPITAL
    self.SMALL_BLIND = SMALL_BLIND
    self.BIG_BLIND = BIG_BLIND
    self.N_PLAYERS = 6

  def run_game(self, players):
    if len(players) != self.N_PLAYERS:
      raise ValueError('Only {} players allowed'.format(self.N_PLAYERS))

    cards = np.tile(np.arange(52), (self.BATCH_SIZE, 1))
    for i in range(self.BATCH_SIZE):
      cards[i, :] = FULL_DECK[np.random.permutation(cards[i, :])]
    community_cards = cards[:, :5]
    hole_cards = np.reshape(cards[:, 5:5 + 2 * len(players)], (self.BATCH_SIZE, len(players), 2))

    with open("tmp_cards.dump", "wb") as f:
      pickle.dump((community_cards, hole_cards), f)

    with open("tmp_cards.dump", "rb") as f:
      community_cards, hole_cards = pickle.load(f)

    folded = np.zeros((self.BATCH_SIZE, len(players)), dtype=bool)
    prev_round_investment = np.zeros((self.BATCH_SIZE, len(players)), dtype=int)

    for player in players:
      player.start_game(self.BATCH_SIZE, self.INITIAL_CAPITAL, self.N_PLAYERS)

    print("PREFLOP")

    # Pre-flop
    bets = self.run_round(players, prev_round_investment, folded, PRE_FLOP, hole_cards, community_cards[:, :0])
    prev_round_investment += bets

    print("FLOP")

    # Flop
    bets = self.run_round(players, prev_round_investment, folded, FLOP, hole_cards, community_cards[:, :3])
    prev_round_investment += bets

    print("TURN")

    # Turn
    bets = self.run_round(players, prev_round_investment, folded, TURN, hole_cards, community_cards[:, :4])
    prev_round_investment += bets

    print("RIVER")

    # River
    bets = self.run_round(players, prev_round_investment, folded, RIVER, hole_cards, community_cards)
    prev_round_investment += bets

    print("SHOWDOWN")

    # Showdown
    pool = np.sum(prev_round_investment, axis=1)
    total_winnings = np.zeros((self.BATCH_SIZE, self.N_PLAYERS), dtype=int)

    hand_scores = self.evaluate_hands(community_cards, hole_cards, np.logical_not(folded))

    ranks = np.argsort(hand_scores, axis=1)
    sorted_hands = np.take_along_axis(hand_scores, indices=ranks, axis=1)
    # Get everyone who has the next best hand and among which pots will be split
    participants = hand_scores == sorted_hands[:, 0][:, None]
    # Get the number of times each pot will be split
    n_splits_per_game = participants.sum(axis=1)
    # Split and distribute the money
    gains = pool // n_splits_per_game
    total_winnings += participants * gains[:, None]

    total_winnings -= prev_round_investment

    return total_winnings

  def run_round(self, players, prev_round_investment, folded, round, hole_cards, community_cards):
    current_bets = np.zeros((self.BATCH_SIZE, self.N_PLAYERS), dtype=int)
    max_bets = np.zeros(self.BATCH_SIZE, dtype=int)
    min_raise = np.zeros(self.BATCH_SIZE, dtype=int)
    min_raise[:] = self.BIG_BLIND
    last_raiser = np.zeros(self.BATCH_SIZE, dtype=int)

    if round == PRE_FLOP:
      current_bets[:, 0] = self.SMALL_BLIND
      current_bets[:, 1] = self.BIG_BLIND
      max_bets[:] = self.BIG_BLIND

    round_countdown = np.zeros(self.BATCH_SIZE, dtype=int)
    round_countdown[:] = self.N_PLAYERS

    while True:
      running_games = np.nonzero(round_countdown > 0)[0]
      for player_idx, player in enumerate(players):
        actions, amounts = player.act(player_idx, round, current_bets, min_raise, prev_round_investment, folded,
                                      last_raiser, hole_cards[:, player_idx, :], community_cards)

        round_countdown[running_games] -= 1

        actions[folded[:, player_idx] == 1] = FOLD

        # print("Player", player_idx)
        # print("Actions", actions)

        ###########
        # CALLING #
        ###########

        # Handle the real checks
        calls = np.where(actions == CALL)[0]
        if calls.size > 0:
          # print("True calls", calls)
          investment = max_bets[calls]
          # Reset the bets and countdown
          max_bets[calls] = investment
          current_bets[calls, player_idx] = investment

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
          # print("False raises", false_raises)
          # All of these players can't afford to raise properly, so they do a false raise and all-in
          actions[false_raises] = CALL
          investment = self.INITIAL_CAPITAL - prev_round_investment[false_raises, player_idx]
          max_bets[false_raises] = np.maximum(investment, max_bets[false_raises])
          current_bets[false_raises, player_idx] = investment

        # Handle the real raises
        raises = np.where(actions == RAISE)[0]
        if raises.size > 0:
          # print("True raises", raises, amounts[raises])
          investment = np.maximum(current_bets[raises, player_idx] + amounts[raises], max_bets[raises] + min_raise[raises])
          investment = np.minimum(investment, self.INITIAL_CAPITAL - prev_round_investment[raises, player_idx])
          assert((investment + prev_round_investment[raises, player_idx] <= self.INITIAL_CAPITAL).all())
          # Reset the bets and countdown
          max_bets[raises] = investment
          current_bets[raises, player_idx] = investment
          round_countdown[raises] = len(players)
        last_raiser[raises] = player_idx

        ###########
        # FOLDING #
        ###########

        folded[np.where(np.logical_and(round_countdown > 0, actions == FOLD))[0], player_idx] = 1
        round_countdown[folded.sum(axis=1) == self.N_PLAYERS] = 0

        # print("Bets after turn", current_bets[:, player_idx])

        if np.sum(round_countdown[running_games]) <= 0:
          return current_bets


  def evaluate_hands(self, community_cards, hole_cards, contenders):
    evaluator = treys.Evaluator()
    # 7463 = 1 lower than the lowest score a hand can have (scores are descending to 1)
    results = np.full((self.BATCH_SIZE, self.N_PLAYERS), 7463)
    for game_idx,community in enumerate(community_cards):
      for player_idx,hole in enumerate(hole_cards[game_idx]):
        if contenders[game_idx, player_idx]:
          results[game_idx, player_idx] = evaluator.evaluate(community.tolist(), hole.tolist())
    return results
