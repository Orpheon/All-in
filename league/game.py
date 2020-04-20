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

    folded = np.zeros((self.BATCH_SIZE, len(players)))
    allined = np.zeros((self.BATCH_SIZE, len(players)))
    prev_round_investment = np.zeros((self.BATCH_SIZE, len(players)))
    allin_poolsize = np.full((self.BATCH_SIZE, self.N_PLAYERS), self.INITIAL_CAPITAL * self.N_PLAYERS)

    for player in players:
      player.start_game(self.BATCH_SIZE, self.INITIAL_CAPITAL, self.N_PLAYERS)

    print("PREFLOP")

    # Pre-flop
    bets, allin_poolsize = self.run_round(players, prev_round_investment, folded, allined, allin_poolsize, PRE_FLOP,
                                          hole_cards, community_cards[:, :0])
    prev_round_investment += bets
    assert(np.equal(folded, allined).sum() == 0)

    print("FLOP")

    # Flop
    bets, allin_poolsize = self.run_round(players, prev_round_investment, folded, allined, allin_poolsize, FLOP,
                                          hole_cards, community_cards[:, :3])
    prev_round_investment += bets
    assert (np.equal(folded, allined).sum() == 0)

    print("TURN")

    # Turn
    bets, allin_poolsize = self.run_round(players, prev_round_investment, folded, allined, allin_poolsize, TURN,
                                          hole_cards, community_cards[:, :4])
    prev_round_investment += bets
    assert (np.equal(folded, allined).sum() == 0)

    print("RIVER")

    # River
    bets, allin_poolsize = self.run_round(players, prev_round_investment, folded, allined, allin_poolsize, RIVER,
                                          hole_cards, community_cards)
    prev_round_investment += bets
    assert (np.equal(folded, allined).sum() == 0)

    print("SHOWDOWN")

    # Showdown
    pool = np.sum(prev_round_investment, axis=1)
    total_winnings = np.zeros((self.BATCH_SIZE, self.N_PLAYERS))

    went_allin = np.zeros((self.BATCH_SIZE, self.N_PLAYERS))
    went_allin[allin_poolsize < self.INITIAL_CAPITAL * self.N_PLAYERS] = 1
    pots = np.minimum(pool[:, None], allin_poolsize)
    hand_scores = self.evaluate_hands(community_cards, hole_cards, np.logical_not(folded))

    # Now we face the actually not that easy task of splitting the pool by winners
    # Take the first rank, and then having a boolean (BATCH_SIZE, N_PLAYERS) array be set to 1 for all of all elements
    # in the sorted scores that are equal to first rank. Sum this along players axis, and you know how often you need to
    # split the pot. Where this gets fun is if one of the players paid in a allin sidepool
    # Find the minimum in the allinpool where that array == 1. Split this pot. Reduce all players'  of the row pot by
    # this amount, set that player's hand_scores and presence in the boolean participation array to nil, and repeat.
    # If some pots are still containing something after this, redo starting from the argsort. As the hands are reset
    # people should automatically get pushed to the back
    while pots.any():
      ranks = np.argsort(hand_scores, axis=1)
      sorted_hands = np.take_along_axis(hand_scores, indices=ranks, axis=1)
      # Get everyone who has the next best hand and among which pots will be split
      participants = hand_scores == sorted_hands[:, 0][:, None]
      # Get the number of times each pot will be split
      n_splits_per_game = participants.sum(axis=1)
      # Get the smallest pot, which will be split this round (ignoring those that have already dropped to 0)
      smallest_pots = np.min(pots, axis=1, initial=self.INITIAL_CAPITAL * self.N_PLAYERS + 1, where=(pots > 0))
      smallest_pots[smallest_pots > self.INITIAL_CAPITAL * self.N_PLAYERS] = 0
      # Add gains according to who won
      gains = smallest_pots / n_splits_per_game
      total_winnings += participants * gains[:, None]
      # Remove those players who have consumed their pools
      players_still_competing = (pots > smallest_pots[:, None])
      hand_scores[players_still_competing] = 7463
      pots = np.maximum(0, pots - smallest_pots[:, None])

    total_winnings -= prev_round_investment

    return total_winnings

  def run_round(self, players, prev_round_investment, folded, allined, allin_poolsize, round, hole_cards, community_cards):
    current_bets = np.zeros((self.BATCH_SIZE, self.N_PLAYERS))
    max_bets = np.zeros(self.BATCH_SIZE)
    min_raise = np.zeros(self.BATCH_SIZE)
    min_raise[:] = self.BIG_BLIND
    current_allin_check = np.zeros((self.BATCH_SIZE, self.N_PLAYERS))
    last_raiser = np.zeros(self.BATCH_SIZE, dtype=int)

    if round == PRE_FLOP:
      current_bets[:, 0] = self.SMALL_BLIND
      current_bets[:, 1] = self.BIG_BLIND
      max_bets[:] = self.BIG_BLIND

    round_countdown = np.zeros(self.BATCH_SIZE)
    round_countdown[:] = self.N_PLAYERS

    while True:
      running_games = np.nonzero(round_countdown > 0)[0]
      for player_idx, player in enumerate(players):
        actions, amounts = player.act(player_idx, round, current_bets, min_raise, prev_round_investment, folded,
                                      allined, last_raiser, hole_cards[:, player_idx, :], community_cards)

        round_countdown[running_games] -= 1

        actions[folded[:, player_idx] == 1] = FOLD
        actions[allined[:, player_idx] == 1] = ALLIN

        # print("Player", player_idx)
        # print("Actions", actions)

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
          actions[false_raises] = ALLIN
          investment = self.INITIAL_CAPITAL - prev_round_investment[false_raises, player_idx]
          max_bets[false_raises] = np.maximum(investment, max_bets[false_raises])
          current_allin_check[false_raises, player_idx] = 1
          current_bets[false_raises, player_idx] = investment

        # Handle the real raises
        raises = np.where(actions == RAISE)[0]
        if raises.size > 0:
          # print("True raises", raises, amounts[raises])
          investment = np.maximum(current_bets[raises, player_idx] + amounts[raises], max_bets[raises] + min_raise[raises])
          # Isolate all the games where someone had just gone allin and save for them the sidepool
          sidepool_indices = np.nonzero(current_allin_check[raises, :])
          # print(current_allin_check)
          # print(sidepool_indices)
          allin_poolsize[sidepool_indices] = (np.sum(prev_round_investment, axis=1) + np.sum(current_bets, axis=1))[sidepool_indices[0]]
          # print(np.sum(prev_round_investment, axis=1))
          # print(np.sum(current_bets, axis=1))
          current_allin_check[raises, :] = 0
          # Reset the bets and countdown
          max_bets[raises] = investment
          current_bets[raises, player_idx] = investment
          round_countdown[raises] = len(players)
        last_raiser[raises] = player_idx

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
          # print("False calls", false_calls)
          # All of these players can't actually afford to call, so they go all-in instead
          actions[false_calls] = ALLIN
          investment = self.INITIAL_CAPITAL - prev_round_investment[false_calls, player_idx]
          current_allin_check[false_calls, player_idx] = 1
          current_bets[false_calls, player_idx] = investment

        # Handle the real checks
        calls = np.where(actions == CALL)[0]
        if calls.size > 0:
          # print("True calls", calls)
          investment = max_bets[calls]
          # Reset the bets and countdown
          max_bets[calls] = investment
          current_bets[calls, player_idx] = investment

        ###########
        # FOLDING #
        ###########

        folded[np.where(np.logical_and(round_countdown > 0, actions == FOLD))[0], player_idx] = 1
        allined[np.where(np.logical_and(round_countdown > 0, actions == ALLIN))[0], player_idx] = 1
        round_countdown[(folded + allined).sum(axis=1) == self.N_PLAYERS] = 0

        # print("Bets after turn", current_bets[:, player_idx])

        if np.sum(round_countdown[running_games]) <= 0:
          return current_bets, allin_poolsize


  def evaluate_hands(self, community_cards, hole_cards, contenders):
    evaluator = treys.Evaluator()
    # 7463 = 1 lower than the lowest score a hand can have (scores are descending to 1)
    results = np.full((self.BATCH_SIZE, self.N_PLAYERS), 7463)
    for game_idx,community in enumerate(community_cards):
      for player_idx,hole in enumerate(hole_cards[game_idx]):
        if contenders[game_idx, player_idx]:
          results[game_idx, player_idx] = evaluator.evaluate(community.tolist(), hole.tolist())
    return results
