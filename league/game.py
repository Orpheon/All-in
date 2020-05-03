import numpy as np
import pickle
import treys

import constants

FULL_DECK = np.array(treys.Deck.GetFullDeck())


class GameEngine:
  def __init__(self, BATCH_SIZE, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND, logger):
    self.BATCH_SIZE = BATCH_SIZE
    self.INITIAL_CAPITAL = INITIAL_CAPITAL
    self.SMALL_BLIND = SMALL_BLIND
    self.BIG_BLIND = BIG_BLIND
    self.logger = logger
    self.N_PLAYERS = 6

  def generate_cards(self):
    cards = np.tile(np.arange(52), (self.BATCH_SIZE, 1))
    for i in range(self.BATCH_SIZE):
      cards[i, :] = FULL_DECK[np.random.permutation(cards[i, :])]
    community_cards = cards[:, :5]
    hole_cards = np.reshape(cards[:, 5:5 + 2 * self.N_PLAYERS], (self.BATCH_SIZE, self.N_PLAYERS, 2))
    return community_cards, hole_cards

  def run_game(self, players):
    self.logger.log(constants.EV_START_NEW_GAME, (self.N_PLAYERS, self.BATCH_SIZE))
    if len(players) != self.N_PLAYERS:
      raise ValueError('Only {} players allowed'.format(self.N_PLAYERS))

    community_cards, hole_cards = self.generate_cards()
    self.logger.log(constants.EV_DEALT_CARDS, (community_cards, hole_cards))

    folded = np.zeros((self.BATCH_SIZE, len(players)), dtype=bool)
    prev_round_investment = np.zeros((self.BATCH_SIZE, len(players)), dtype=int)

    for player in players:
      player.start_game(self.BATCH_SIZE, self.INITIAL_CAPITAL, self.N_PLAYERS)

    # Pre-flop
    bets, _ = self.run_round(players, prev_round_investment, folded, constants.PRE_FLOP, hole_cards, community_cards[:, :0])
    prev_round_investment += bets

    # Flop
    bets, _ = self.run_round(players, prev_round_investment, folded, constants.FLOP, hole_cards, community_cards[:, :3])
    prev_round_investment += bets

    # Turn
    bets, _ = self.run_round(players, prev_round_investment, folded, constants.TURN, hole_cards, community_cards[:, :4])
    prev_round_investment += bets

    # River
    bets, end_state = self.run_round(players, prev_round_investment, folded, constants.RIVER, hole_cards, community_cards)
    prev_round_investment += bets

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

    self.logger.log(constants.EV_END_GAME, (ranks, total_winnings))
    self.logger.save_to_file()

    for player_idx, player in enumerate(players):
      player.end_trajectory(*end_state, total_winnings[:, player_idx])

    return total_winnings

  def run_round(self, players, prev_round_investment, folded, round, hole_cards, community_cards):
    """
    :param players: [Player]
    :param prev_round_investment: np.ndarray(batchsize, n_players) = int
    :param folded: np.ndarray(batchsize, n_players) = bool
    :param round: int ∈ {0..3}
    :param hole_cards: np.ndarray(batchsize, n_players, 2) = treys.Card
    :param community_cards: np.ndarray(batchsize, n_players, {0,3,4,5}) = treys.Card
    :return: current_bets: np.ndarray(batchsize, n_players)=int {0-200}
    """
    current_bets = np.zeros((self.BATCH_SIZE, self.N_PLAYERS), dtype=int)
    max_bets = np.zeros(self.BATCH_SIZE, dtype=int)
    min_raise = np.zeros(self.BATCH_SIZE, dtype=int)
    min_raise[:] = self.BIG_BLIND
    last_raiser = np.zeros(self.BATCH_SIZE, dtype=int)

    player_order = list(enumerate(players))

    if round == constants.PRE_FLOP:
      current_bets[:, 0] = self.SMALL_BLIND
      current_bets[:, 1] = self.BIG_BLIND
      max_bets[:] = self.BIG_BLIND
      player_order = player_order[2:] + player_order[:2]

    round_countdown = np.zeros(self.BATCH_SIZE, dtype=int)
    round_countdown[:] = self.N_PLAYERS

    while True:
      running_games = np.nonzero(round_countdown > 0)[0]

      for player_idx, player in player_order:
        actions, amounts = player.act(player_idx, round, current_bets, min_raise, prev_round_investment, folded,
                                      last_raiser, hole_cards[:, player_idx, :], community_cards)
        # Disabled when not necessary because it bloats the log size (by ~500 kB or so, which triples the size)
        self.logger.log(constants.EV_PLAYER_ACTION, (round, player_idx, actions, amounts, round_countdown, folded[:, player_idx]))

        # People who have already folded continue to fold
        actions[folded[:, player_idx] == 1] = constants.FOLD
        # People who have gone all-in continue to be all-in
        actions[prev_round_investment[:, player_idx] + current_bets[:, player_idx] == self.INITIAL_CAPITAL] = constants.CALL

        ###########
        # CALLING #
        ###########

        calls = np.where(np.logical_and(round_countdown > 0, actions == constants.CALL))[0]
        if calls.size > 0:
          # print("True calls", calls)
          investment = max_bets[calls]
          # Reset the bets and countdown
          max_bets[calls] = investment
          current_bets[calls, player_idx] = investment

        ###########
        # RAISING #
        ###########

        raises = np.where(np.logical_and(round_countdown > 0, actions == constants.RAISE))[0]
        if raises.size > 0:
          # print("True raises", raises, amounts[raises])
          investment = np.maximum(current_bets[raises, player_idx] + amounts[raises], max_bets[raises] + min_raise[raises])
          investment = np.minimum(investment, self.INITIAL_CAPITAL - prev_round_investment[raises, player_idx])
          assert((investment + prev_round_investment[raises, player_idx] <= self.INITIAL_CAPITAL).all())
          # Reset the bets and countdown
          max_bets[raises] = investment
          current_bets[raises, player_idx] = investment
          round_countdown[raises] = self.N_PLAYERS
          last_raiser[raises] = player_idx

        ###########
        # FOLDING #
        ###########

        folded[np.where(np.logical_and(round_countdown > 0, actions == constants.FOLD))[0], player_idx] = 1
        round_countdown[running_games] -= 1
        #TODO: if all folded stops game, improves performance but breaks tests
        # round_countdown[folded.sum(axis=1) == self.N_PLAYERS-1] = 0

        if np.max(round_countdown[running_games]) <= 0:
          return current_bets, (player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser,
                                hole_cards[:, player_idx, :], community_cards)

  def evaluate_hands(self, community_cards, hole_cards, contenders):
    evaluator = treys.Evaluator()
    # 7463 = 1 lower than the lowest score a hand can have (scores are descending to 1)
    results = np.full((self.BATCH_SIZE, self.N_PLAYERS), 7463)
    for game_idx,community in enumerate(community_cards):
      for player_idx,hole in enumerate(hole_cards[game_idx]):
        if contenders[game_idx, player_idx]:
          results[game_idx, player_idx] = evaluator.evaluate(community.tolist(), hole.tolist())
    return results
