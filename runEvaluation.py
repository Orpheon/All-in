import time
import argparse
import importlib
import json
import os
import shutil
import pickle
import numpy as np
import treys

N_TESTCASES = 10000

from baseline.ConsolePokerPlayer import ConsolePlayer
from configuration.CashGameConfig import CashGameConfig
from agent.MyBot import MyBotPlayer
from agent.LeagueTestBot import LeagueTestBot
from agent.random.RandomAgentPyPoker import RandomAgent

parser = argparse.ArgumentParser(description='Run poker evaluation')
parser.add_argument('--config', help='Config file')
parser.add_argument('--store_result', dest='store_result', action='store_true')
parser.set_defaults(store_result=False)


def load_configuration(config_file: str):
  with open(config_file, 'r', encoding='utf-8') as fp:
    config = json.load(fp)
  return config


def register_players(config_file, poker_config) -> None:
  """
  Seat baseline and players set in the evaluation_config.json file.
  :param config_file: json
  :param poker_config: CashGame configuration object
  """
  for module, cls in config_file['baselines'].items():
    module = importlib.import_module(f'baseline.{module}')
    baseline_class = getattr(module, cls)
    poker_config.register_player(cls, baseline_class())

  for module, cls in config_file['players'].items():
    module = importlib.import_module(f'agent.{module}')
    baseline_class = getattr(module, cls)
    poker_config.register_player(cls, baseline_class())


if __name__ == '__main__':
  args = parser.parse_args()

  if args.config and os.path.isfile(args.config):
    # Used by run_evaluation.sh script
    config_file = load_configuration(args.config)
    poker_config = CashGameConfig(evaluations=config_file['n_evaluations'],
                    small_blind_amount=config_file['small_blind'],
                    log_file_location=config_file['log_file_location'])
    register_players(config_file, poker_config)
  else:
    # Use this for manual evaluation
    poker_config = CashGameConfig(evaluations=N_TESTCASES, log_file_location="leaguetestlog.json")
    poker_config.register_player("RandomAgent", RandomAgent(1, ''))
    poker_config.register_player("RandomAgent", RandomAgent(2, ''))
    poker_config.register_player("RandomAgent", RandomAgent(3, ''))
    poker_config.register_player("RandomAgent", RandomAgent(4, ''))
    poker_config.register_player("RandomAgent", RandomAgent(5, ''))
    poker_config.register_player("RandomAgent", RandomAgent(6, ''))
    # poker_config.add_all_available_baselines()

  print(f"Start evaluating {poker_config.evaluations} hands")
  start = time.time()
  result = poker_config.run_evaluation()
  end = time.time()
  time_taken = end - start
  print('Evaluation Time: ', time_taken)
  print("Compressing files..")

  community_cards = np.ndarray((N_TESTCASES, 5), dtype=int)
  hole_cards = np.ndarray((N_TESTCASES, 6, 2), dtype=int)
  gains = np.ndarray((N_TESTCASES, 6), dtype=int)

  # with open("leaguetestlog.json", "r") as f:
  #   results = json.load(f)
  # # Apparently stuff is sorted by keys here, I do not have the slightest idea why
  #
  # with open("leaguetestlog_readable.json", "w") as f:
  #   json.dump(results, f, sort_keys=True, indent=2)
  #
  # for k,v in results.items():



  # old_cashgame = {}

  # for round_idx, filepath in enumerate(filelist):
  #   with open(os.path.join("testcases", filepath), "rb") as f:
  #     data = pickle.load(f)
  #     players = data['seats']
  #     # Seat rotation
  #     players = players[data['small_blind_pos']:] + players[:data['small_blind_pos']]
  #     # Log items of interest
  #     community_cards[round_idx] = np.array([treys.Card.new(x) for x in data['community_cards']])
  #     for player_idx, player in enumerate(players):
  #       hole_cards[round_idx, player_idx] = np.array([treys.Card.new(x) for x in player['hole_card']])
  #       if player['uuid'] in old_cashgame:
  #         gains[round_idx - 1, player_idx] = player['cashgame_stack'] - old_cashgame[player['uuid']]
  #       old_cashgame[player['uuid']] = player['cashgame_stack']
  #
  #     # print(json.dumps(data, sort_keys=True, indent=2))
  #     # print(old_cashgame)
  #     # input()
  # print(gains.sum(axis=0))
  # # with open("testcases.pickle", "wb") as f:
  # #   pickle.dump(data, f)
  # shutil.rmtree("testcases")


  if args.store_result:
    with open('result.txt', 'w+') as fp:
      pretty_result = '\n'.join(f"{rank + 1:2}. Player: {name:>25}, Stack: {stack:n}" for rank, (name, stack) in
                    enumerate(result.items()))
      fp.write(pretty_result)
