import numpy as np
import treys
import random
import os

import constants

FULL_DECK = np.array(treys.Deck.GetFullDeck())
def generate_cards(N_PLAYERS, BATCH_SIZE):
  cards = np.tile(np.arange(52), (BATCH_SIZE, 1))
  for i in range(BATCH_SIZE):
    cards[i, :] = FULL_DECK[np.random.permutation(cards[i, :])]
  community_cards = cards[:, :5]
  hole_cards = np.reshape(cards[:, 5:5 + 2 * N_PLAYERS], (BATCH_SIZE, N_PLAYERS, 2))
  return community_cards, hole_cards

def compute_strat_vector(agent, agent_name):
  # Mutable
  N_CARD_PERCENTILE_BINS = 10
  N_TOTAL_POT_BINS = 8
  N_OWN_POT_SIZE_BINS = 6
  N_MIN_SAMPLES_PER_BIN = 25
  BATCH_SIZE = 10000
  INITIAL_CAPITAL = 200
  MIN_RAISE = 4
  ACTION_SPACE_BINS = INITIAL_CAPITAL // 4 + 1

  # Fixed
  N_ROUNDS = 4
  N_SEATING_POSITIONS = 6
  N_MAX_ACTIVE_PLAYERS = 5
  N_PLAYERS_TOTAL = 6

  path_root = "strategy"

  n_games = np.zeros((
    N_SEATING_POSITIONS,
    N_ROUNDS,
    N_MAX_ACTIVE_PLAYERS,
    N_OWN_POT_SIZE_BINS,
    N_TOTAL_POT_BINS,
    N_CARD_PERCENTILE_BINS
  ))
  strategy = np.zeros((
    N_SEATING_POSITIONS,
    N_ROUNDS,
    N_MAX_ACTIVE_PLAYERS,
    N_OWN_POT_SIZE_BINS,
    N_TOTAL_POT_BINS,
    N_CARD_PERCENTILE_BINS,
    ACTION_SPACE_BINS
  ))

  agent.initialize(BATCH_SIZE, INITIAL_CAPITAL, N_PLAYERS_TOTAL)

  print("Generating strategy vector for {0} in {1}-dimensional space..".format(agent_name, n_games.size))

  ranktable = np.stack((
    np.load(os.path.join(path_root, "5card_rank_percentile.npy")),
    np.load(os.path.join(path_root, "6card_rank_percentile.npy")),
    np.load(os.path.join(path_root, "7card_rank_percentile.npy"))
  ))

  preflop_table = np.load(os.path.join(path_root, "preflop_ranks.npy")).tolist()

  evaluator = treys.Evaluator()

  lowest_bin_count = np.amin(n_games)
  while lowest_bin_count <= N_MIN_SAMPLES_PER_BIN:
    print("Starting filling round")
    print("\tLowest number currently:", lowest_bin_count)
    print("\tNumber of bins at lowest:", np.count_nonzero(n_games == lowest_bin_count))
    print("\tAverage bin count:", np.mean(n_games))
    for round in range(0, N_ROUNDS):
      n_community_cards = [0, 3, 4, 5][round]
      for seating_position in range(N_SEATING_POSITIONS):
        for n_active_players in range(2, N_MAX_ACTIVE_PLAYERS):
          for own_pot_idx in range(N_OWN_POT_SIZE_BINS):
            own_pot = int(INITIAL_CAPITAL * own_pot_idx / N_OWN_POT_SIZE_BINS)
            community_cards, all_hole_cards = generate_cards(N_PLAYERS_TOTAL, BATCH_SIZE)
            hole_cards = all_hole_cards[:, seating_position, :]
            rank_bin = np.zeros(BATCH_SIZE)

            folded_players = np.zeros((BATCH_SIZE, N_PLAYERS_TOTAL - n_active_players), dtype=int)
            active_players = np.zeros((BATCH_SIZE, n_active_players), dtype=int)
            last_raiser = np.zeros(BATCH_SIZE, dtype=int)
            for i in range(BATCH_SIZE):
              folded_row = random.sample([x for x in range(N_PLAYERS_TOTAL) if x != seating_position], N_PLAYERS_TOTAL - n_active_players)
              active_row = [x for x in range(N_PLAYERS_TOTAL) if x not in folded_row]
              folded_players[i, :] = folded_row
              active_players[i, :] = active_row
              last_raiser[i] = random.choice(active_row)
              if round != constants.PRE_FLOP:
                rank = evaluator.evaluate(community_cards[i, :n_community_cards].tolist(), hole_cards[i].tolist())
                percentile = 1 - ranktable[round - constants.FLOP, rank]
                rank_bin[i] = int(percentile * N_CARD_PERCENTILE_BINS)
              else:
                sorted_hole = sorted(hole_cards[i].tolist())
                percentile = preflop_table.index(sorted_hole) / len(preflop_table)
                rank_bin[i] = int(percentile * N_CARD_PERCENTILE_BINS)

            current_bets = np.zeros((BATCH_SIZE, N_PLAYERS_TOTAL))
            prev_round_investment = np.zeros((BATCH_SIZE, N_PLAYERS_TOTAL))

            if round != constants.PRE_FLOP:
              portions = np.random.random(size=(BATCH_SIZE, round + 1))
              # If we are the last raiser, then we raised in the last round, which means there are no bets standing this round
              portions[last_raiser == seating_position, -1] = 0
              portions /= np.sum(portions, axis=1)[:, None]

              self_prev_investment = (np.sum(portions[:, :-1], axis=1) * own_pot)[:, None]
              self_current_investment = (portions[:, -1] * own_pot)[:, None]

              prev_round_investment[
                np.arange(prev_round_investment.shape[0])[:, None], folded_players
              ] += np.random.random(size=folded_players.shape) * self_prev_investment

              # https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops
              prev_round_investment[np.arange(prev_round_investment.shape[0])[:, None], active_players] += self_prev_investment
              current_bets[np.arange(current_bets.shape[0])[:, None], active_players] += self_current_investment

              # small blind + big blind
              prev_round_investment[:, 0] = np.maximum(prev_round_investment[:, 0], 2)
              prev_round_investment[:, 1] = np.maximum(prev_round_investment[:, 1], 4)
            else:
              # https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops
              current_bets[np.arange(current_bets.shape[0])[:, None], folded_players] += np.random.random(size=folded_players.shape) * own_pot
              current_bets[np.arange(current_bets.shape[0])[:, None], active_players] += own_pot


            _, last_raiser_col = np.where(active_players == last_raiser[:, None])
            _, seating_position_col = np.where(active_players == seating_position)
            for i in range(BATCH_SIZE):
              if last_raiser_col[i] < seating_position_col[i]:
                for active_player in active_players[i, last_raiser_col[i]:seating_position_col[i]]:
                  current_bets[i, active_player] += MIN_RAISE
              elif last_raiser_col[i] > seating_position_col[i]:
                for active_player in active_players[i, last_raiser_col[i]:]:
                  current_bets[i, active_player] += MIN_RAISE
                for active_player in active_players[i, :seating_position_col[i]]:
                  current_bets[i, active_player] += MIN_RAISE

            folded_1h = np.zeros((BATCH_SIZE, N_PLAYERS_TOTAL))
            folded_1h[folded_players] = 1

            no_bet_rounds = np.where(~current_bets.any(axis=1))[0]
            last_raiser[no_bet_rounds] = np.argmax(prev_round_investment[no_bet_rounds, :], axis=1)

            actions, amounts = agent.act(
              player_idx=seating_position,
              round=round,
              active_games=np.ones(BATCH_SIZE),
              current_bets=current_bets,
              min_raise=np.full(BATCH_SIZE, MIN_RAISE),
              prev_round_investment=prev_round_investment,
              folded=folded_1h,
              last_raiser=last_raiser,
              hole_cards=hole_cards,
              community_cards=community_cards[:, :n_community_cards]
            )

            # print("Seating position", seating_position)
            # print("Round", round)
            # print("Active players", active_players[5])
            # print("Current bets", current_bets[5, :])
            # print("Prev round investment", prev_round_investment[5, :])
            # print("Last raiser", last_raiser[5])
            # print("Card rank", rank_bin[5])
            # print("Action", actions[5])
            # print("Amount", amounts[5])
            # input()

            amounts[actions == constants.CALL] = MIN_RAISE
            amounts[np.logical_and(actions == constants.CALL, last_raiser == seating_position)] = 0
            action_bins = np.floor(amounts * ACTION_SPACE_BINS / (INITIAL_CAPITAL+1)).astype(int)
            action_bins[actions == constants.FOLD] = 0

            total_pot_idx = np.floor(np.sum(current_bets + prev_round_investment, axis=1) * N_TOTAL_POT_BINS / N_PLAYERS_TOTAL / (INITIAL_CAPITAL + 1)).astype(int)

            for k in range(N_CARD_PERCENTILE_BINS):
              n_games[seating_position, round, n_active_players, own_pot_idx, total_pot_idx, k] += np.count_nonzero(rank_bin == k)
              strategy[seating_position, round, n_active_players, own_pot_idx, total_pot_idx, k, action_bins] += np.count_nonzero(rank_bin == k)

    lowest_bin_count = np.amin(n_games)

  print("Strategy vector for {0} finished, min {1}, max {2} games.".format(agent_name, np.amin(n_games), np.amax(n_games)))
  print("Saving..")
  os.makedirs(os.path.join(path_root, "strat_vectors"), exist_ok=True)
  np.save(os.path.join(path_root, "strat_vectors", agent_name+"_strategy.npy"), strategy)
  print("Saved.")

  return strategy
