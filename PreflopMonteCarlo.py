import json
from time import time

from agent.allin.allinAgentNP import AllinAgentNP
from league.game import GameEngine
from league.logger import GenericLogger
from logPlotter import LogfileCollector


def save_winrates_to_file(winrates):
  with open('preflop_monte_carlo_winrates.json', 'w') as f:
    f.write(json.dumps(winrates))


if __name__ == '__main__':
  BATCH_SIZE = 100_000
  INITIAL_CAPITAL = 200
  SMALL_BLIND = 1
  BIG_BLIND = 2

  N_RUNS = 1000

  #logger = GenericLogger()
  #game_engine = GameEngine(BATCH_SIZE, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND, logger)

  #matchup = [AllinAgentNP(None, None) for _ in range(6)]
  #st1_time = time()

  #for i in range(N_RUNS):
  #  print(i, N_RUNS)
  #  game_engine.run_game(matchup)

  st2_time = time()
  #print('match time', st2_time - st1_time)

  logFileCollector = LogfileCollector(output_folder=None)
  logFileCollector.load_n_most_recent_logfiles(N_RUNS)
  winrate_per_hole_hand = logFileCollector.winrate_per_hole_hand
  print(winrate_per_hole_hand)
  #TODO: commented out to protect file from unintentional editing
  #save_winrates_to_file([[[int(c) for c in cards], wins[0] / wins[1] if wins[1] != 0 else 0.0]
  #                       for cards, wins in winrate_per_hole_hand.items()])

  print('file compile time', time() - st2_time)