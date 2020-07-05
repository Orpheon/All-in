import numpy as np
import treys
import random

import constants

# Dimensions
#   Card value: 1 - 7462
#     linear? log? - n bins
#   Own pot investment - m bins
#   Pot investment of second biggest player - o bins
#   Pot investment of third biggest player - p bins
#   Pot investment of fourth biggest player - q bins
#   Pot investment of fifth biggest player - r bins
#   Pot investment of last player - s bins
#   Seating position - 6
#   Round - 4

# m = o = p = q = r = s
#   = 4
#     ==> total = 98'304 * n bins
#   = 6
#     ==> total = 1'119'744 * n bins

# Different approach: either players are in the running, or they're not
# instead of comparing each player pot investment, compare total pot from players who are out,
# and how many players are still in
#   Own pot investment - m bins
#   Total pot size - u bins
#   players still in - 5
#     m = 6, u = 8
#       ==> total = 5760 * n bins (more or less, minus invalid states)

# Card value: Use percentile hand rank of that number of cards
#   different distribution per round type
#   for preflop: use percentile preflop quality table from the other monte carlo

FULL_DECK = np.array(treys.Deck.GetFullDeck())
def generate_cards(N_PLAYERS, BATCH_SIZE):
  cards = np.tile(np.arange(52), (BATCH_SIZE, 1))
  for i in range(BATCH_SIZE):
    cards[i, :] = FULL_DECK[np.random.permutation(cards[i, :])]
  community_cards = cards[:, :5]
  hole_cards = np.reshape(cards[:, 5:5 + 2 * N_PLAYERS], (BATCH_SIZE, N_PLAYERS, 2))
  return community_cards, hole_cards

def compute_strat_vector(agent):
  # Mutable
  N_CARD_PERCENTILE_BINS = 10
  N_TOTAL_POT_BINS = 8
  N_OWN_POT_SIZE_BINS = 6
  N_MIN_SAMPLES_PER_BIN = 25
  BATCH_SIZE = 10000
  MIN_RAISE = 4 / 200
  # TODO: Action space
  ACTION_SPACE_BINS = 10

  # Fixed
  N_ROUNDS = 4
  N_SEATING_POSITIONS = 6
  N_ACTIVE_PLAYERS = 5
  N_PLAYERS_TOTAL = 6

  n_games = np.ndarray((
    N_SEATING_POSITIONS,
    N_ROUNDS,
    N_ACTIVE_PLAYERS,
    N_OWN_POT_SIZE_BINS,
    N_TOTAL_POT_BINS,
    N_CARD_PERCENTILE_BINS
  ))
  strategy = np.ndarray((
    N_SEATING_POSITIONS,
    N_ROUNDS,
    N_ACTIVE_PLAYERS,
    N_OWN_POT_SIZE_BINS,
    N_TOTAL_POT_BINS,
    N_CARD_PERCENTILE_BINS,
    ACTION_SPACE_BINS
  ))

  print("Generating strategy vector for {0} in {1}-dimensional space..".format(str(agent), n_games.size))

  ranktable = np.stack((
    np.load("5card_rank_percentile.npy"),
    np.load("5card_rank_percentile.npy"),
    np.load("5card_rank_percentile.npy")
  ))

  evaluator = treys.Evaluator()

  # TODO: While not enough samples
  for round in range(N_ROUNDS):
    n_community_cards = [0, 3, 4, 5][round]
    for seating_position in range(N_SEATING_POSITIONS):
      for unfolded_players in range(1, N_ACTIVE_PLAYERS):
        for own_pot in [i/N_OWN_POT_SIZE_BINS for i in range(N_OWN_POT_SIZE_BINS)]:
          community_cards, all_hole_cards = generate_cards(N_PLAYERS_TOTAL, BATCH_SIZE)
          hole_cards = all_hole_cards[:, seating_position, :]
          rank_bin = np.zeros(BATCH_SIZE)

          folded_players = np.zeros((BATCH_SIZE, N_PLAYERS_TOTAL - N_ACTIVE_PLAYERS))
          active_players = np.zeros((BATCH_SIZE, N_ACTIVE_PLAYERS))
          last_raiser = np.zeros(BATCH_SIZE)
          for i in range(BATCH_SIZE):
            folded_row = random.choice([x for x in range(N_PLAYERS_TOTAL) if x != seating_position], N_PLAYERS_TOTAL - N_ACTIVE_PLAYERS)
            active_row = [x for x in range(N_PLAYERS_TOTAL) if x not in folded_row]
            folded_players[i, :] = folded_row
            active_players[i, :] = active_row
            last_raiser[i] = random.choice(active_row)
            if round != constants.PRE_FLOP:
              rank = evaluator.evaluate(community_cards[i, :n_community_cards].tolist(), hole_cards[i].tolist())
              percentile = 1 - ranktable[rank, round - constants.FLOP]
              rank_bin[i] = int(percentile * N_CARD_PERCENTILE_BINS)
            else:
              pass
              # TODO preflop card ranking

          current_bets = np.zeros((BATCH_SIZE, N_PLAYERS_TOTAL))
          prev_round_investment = np.zeros((BATCH_SIZE, N_PLAYERS_TOTAL))

          if round != constants.PRE_FLOP:
            prev_round_investment[folded_players] = np.random.random(size=folded_players.shape) * own_pot
            portions = np.random.random(size=(BATCH_SIZE, round + 1))
            # If we are the last raiser, then we raised in the last round, which means there are no bets standing this round
            portions[last_raiser == seating_position, -1] = 0
            portions /= np.sum(portions, axis=1)
            prev_round_investment[active_players] = np.sum(portions[:, :-1], axis=1) * own_pot
            current_bets[active_players] = portions[:, -1] * own_pot
          else:
            current_bets[folded_players] = np.random.random(size=folded_players.shape) * own_pot
            current_bets[active_players] = own_pot

          for i in range(BATCH_SIZE):
            if last_raiser[i] < seating_position:
              for active_player in active_players[i, active_players[i].index(last_raiser[i]) : active_players[i].index(seating_position)]:
                current_bets[i, active_player] += MIN_RAISE
            else:
              for active_player in active_players[i, active_players[i].index(seating_position):] + active_players[i, :active_players[i].index(last_raiser[i])]:
                current_bets[i, active_player] += MIN_RAISE

          folded_1h = np.zeros((BATCH_SIZE, N_PLAYERS_TOTAL))
          folded_1h[folded_players] = 1

          actions, amounts = agent.act(
            player_idx=seating_position,
            round=round,
            active_rounds=np.ones(BATCH_SIZE),
            current_bets=current_bets,
            min_raise=np.full(BATCH_SIZE, MIN_RAISE),
            prev_round_investment=prev_round_investment,
            folded=folded_1h,
            last_raiser=last_raiser,
            hole_cards=hole_cards,
            community_cards=community_cards[:, :n_community_cards]
          )
          # TODO: Handle actions and amounts somehow