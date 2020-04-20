import json
import time

import treys

from agent.LeagueTestBot import LeagueTestBot
from configuration.CashGameConfig import CashGameConfig
from league.game import GameEngine
from league.logger import Logger

import numpy as np


#
# helper
#


def id_from_string(card_string):
  return treys.Card.new(card_string)


def cards_ids_from_strings(card_strings):
  return [id_from_string(card_string) for card_string in card_strings]


action_ids = {'FOLD': 0.0, 'CALL': 1.0, 'RAISE': 2.0, 'ALLIN': 3.0}


def action_id_from_str(action_str):
  return action_ids[action_str]


class MockGameEngine(GameEngine):
  def __init__(self, batch_size, initial_capital, small_blind, big_blind, fixed_community_cards, fixed_hole_cards,
               logger):
    super().__init__(batch_size, initial_capital, small_blind, big_blind, logger)
    self._fixed_community_cards = fixed_community_cards
    self._fixed_hole_cards = fixed_hole_cards

  def generate_cards(self):
    return self._fixed_community_cards, self._fixed_hole_cards


class ActionProvider:
  def __init__(self, actions, amounts):
    self.actions = actions
    self.amounts = amounts
    self.idx = -1

  def next_action(self):
    self.idx += 1
    return self.actions[self.idx, :], self.amounts[self.idx, :]


class MockPlayer(LeagueTestBot):
  def __init__(self, action_provider):
    super().__init__()
    self.action_provider = action_provider

  def act(self, player_idx, round, current_bets, min_raise, prev_round_investment, hole_cards, community_cards):
    return self.action_provider.next_action()

  def end_trajectory(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser, hole_cards, community_cards, gains):
    pass

if __name__ == '__main__':
  #
  # config
  #
  N_TESTCASES = 1
  INITIAL_CAPITAL = 200
  SMALL_BLIND = 1
  BIG_BLIND = 2
  N_PLAYERS = 6

  #
  # run pypokerengine as reference
  #
  poker_config = CashGameConfig(evaluations=N_TESTCASES, log_file_location="leaguetestlog.json")
  for _ in range(N_PLAYERS):
    poker_config.register_player("LeagueTestBot", LeagueTestBot())

  print(f"Start evaluating {poker_config.evaluations} hands")
  start = time.time()
  #  result = poker_config.run_evaluation()
  end = time.time()
  time_taken = end - start
  print('Evaluation Time: ', time_taken)

  #
  # import results
  #
  with open("leaguetestlog.json", "r") as f:
    results = json.load(f)

  # Apparently stuff is sorted by keys here, I do not have the slightest idea why
  with open("leaguetestlog_readable.json", "w") as f:
    json.dump(results, f, sort_keys=True, indent=2)

  #
  # transform results for vectorized engine
  #
  rounds_by_dealer_pos = {dealer_pos: [] for dealer_pos in range(N_PLAYERS)}

  seat_actions = [[] for _ in range(N_PLAYERS)]

  sorted_results = sorted([(int(r_idx[6:]) - 1, info) for r_idx, info in results.items()], key=lambda x: x[0])

  community_cards = np.array([[0 for _ in range(5)] for _ in range(N_TESTCASES)])
  hole_hands = np.array([[[0, 0] for _ in range(N_PLAYERS)] for _ in range(N_TESTCASES)])

  seat_actions_counter = {i: 0 for i in range(N_PLAYERS)}

  # calculate community and hole cards
  for round_idx, info in sorted_results:
    first_actor = (info['round_state']['small_blind_pos'])
    seats = [seat['uuid'] for seat in info['round_state']['seats']]
    seats = {uuid: idx for idx, uuid in enumerate(seats[first_actor:] + seats[:first_actor])}
    community_cards[round_idx] = cards_ids_from_strings(info['round_state']['community_card'])
    for seat in info['round_state']['seats']:
      seat_idx = seats[seat['uuid']]
      hole_hands[round_idx][seat_idx] = cards_ids_from_strings(seat['hole_card'])

  # calculate actions
  all_actions = []
  all_amounts = []

  first_actor = [(hand['round_state']['small_blind_pos']) for _, hand in sorted_results]
  seats_each_round = [[seat['uuid'] for seat in hand['round_state']['seats']] for _, hand in sorted_results]
  seats_each_round_fixed = [ser[fa:] + ser[:fa] for fa, ser in zip(first_actor, seats_each_round)]

  for round in ['preflop', 'flop', 'turn', 'river']:
    action_histories = [hand['round_state']['action_histories'].get(round, []) for hand_idx, hand in sorted_results]
    action_histories_no_blinds = [
      [action for action in action_hist if action['action'] != 'SMALLBLIND' and action['action'] != 'BIGBLIND']
      for action_hist in action_histories]
    action_pointer = [0 for _ in sorted_results]
    action_max = [len(act_hist) for act_hist in action_histories_no_blinds]
    agent_idx = 2 if round == 'preflop' else 0

    round_actions = []
    round_amounts = []

    while action_pointer != action_max:

      actions = ['FOLD'] * len(sorted_results)
      amounts = [0] * len(sorted_results)

      for hand_idx, hand in sorted_results:
        if action_pointer[hand_idx] < action_max[hand_idx]:
          action = action_histories_no_blinds[hand_idx][action_pointer[hand_idx]]
          if seats_each_round_fixed[hand_idx][agent_idx] == action['uuid']:
            actions[hand_idx] = action['action']
            amounts[hand_idx] = action.get('paid', 0)
            action_pointer[hand_idx] += 1

      agent_idx = (agent_idx + 1) % N_PLAYERS
      round_actions.append([action_id_from_str(a) for a in actions])
      round_amounts.append(amounts)

    shortest_possible_round = 6
    for _ in range(shortest_possible_round-len(round_actions)):
      actions = ['FOLD'] * len(sorted_results)
      amounts = [0] * len(sorted_results)
      round_actions.append([action_id_from_str(a) for a in actions])
      round_amounts.append(amounts)

    all_actions.extend(round_actions)
    all_amounts.extend(round_amounts)

  all_actions.append([action_id_from_str(i) for i in ['FOLD'] * len(sorted_results)])
  all_amounts.append([0] * len(sorted_results))

  all_actions = np.array(all_actions)
  all_amounts = np.array(all_amounts)

  # calculate winnings
  winnings = np.array([[0.0 for _ in range(N_PLAYERS)] for _ in range(N_TESTCASES)])
  for round_idx, info in sorted_results:
    for seat in info['round_state']['seats']:
      for seat_idx, uuid in enumerate(seats_each_round_fixed[round_idx]):
        if uuid == seat['uuid']:
          winnings[round_idx][seat_idx] = seat['stack'] - 200

  '''
  print('community_cards')
  print(community_cards.shape)
  print('hole_hands')
  print(hole_hands.shape)
  '''
  #print('all_actions')
  #print(all_actions)
  #print('all_amounts')
  #print(all_amounts)
  #print(np.concatenate((all_actions, all_amounts),axis=1))

  #
  # setup test environment
  #
  logger = Logger('testoutput.json')
  mock_game_engine = MockGameEngine(N_TESTCASES, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND, community_cards,
                                    hole_hands, logger)
  action_provider = ActionProvider(all_actions, all_amounts)
  PLAYERS = [MockPlayer(action_provider) for _ in range(N_PLAYERS)]
  rv = mock_game_engine.run_game(PLAYERS)

  for i in range(N_TESTCASES):
    print('pypoke', winnings[i, :])
    print('vector', rv[i, :])

  print(logger.hole_cards)
  print(logger.community_cards)

  print(np.array([[[treys.Card.print_pretty_cards(player)] for player in batch]
                  for batch in logger.hole_cards.tolist()]))
  print(np.array([treys.Card.print_pretty_cards(batch)
                  for batch in logger.community_cards.tolist()]))

  for a in logger.action_history:
    print(a)

  print(logger.still_playing)
