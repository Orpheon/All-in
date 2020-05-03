from typing import Dict, List, Union, Tuple
from agent.random.randomAgentPyPoker import RandomAgent

import json
import pickle
import os
import random

class LeagueTestBot(RandomAgent):
  def __init__(self):
    self._fold_ratio = 0.1
    self._call_ratio = 0.5
    self.decks = []
    self.actions = []
    self.game_actions = []
    self.gains = []

  def __str__(self):
    return "LeagueTestBot"

  def declare_action(self, valid_actions, hole_card, round_state):
    choice = self.__choice_action(valid_actions)
    action = choice["action"]
    amount = choice["amount"]
    if action == "raise":
      amount['max'] = min(amount['max'], 20)
      amount = random.randrange(amount["min"], max(amount["min"], amount["max"]) + 1)
    return action, amount

  def __choice_action(self, valid_actions):
    r = random.random()
    if r <= self._fold_ratio:
      return valid_actions[0]
    elif r <= self._call_ratio:
      return valid_actions[1]
    else:
      return valid_actions[2]

  def receive_game_start_message(self, game_info: Dict[str, Union[int, Dict, List]]) -> None:
    """
    Called once the game started.
    :param game_info: Dictionary containing game rules, # of rounds, initial stack, small blind and players at the table.
    """
    pass

  def receive_round_start_message(self, round_count: int, hole_card: List[str],
                  seats: List[Dict[str, Union[str, int]]]) -> None:
    """
    Called once a round starts.
    :param round_count: Round number, in Cash Game always 1.
    :param hole_card: Cards in possession of the player.
    :param seats: Players at the table.
    """
    pass

  def receive_street_start_message(self, street: str, round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
    """
    Gets called at every stage (preflop, flop, turn, river, showdown).
    :param street: Game stage
    :param round_state: Dictionary containing the round state
    """
    pass

  def receive_game_update_message(self, action: Dict[str, Union[str, int]],
                  round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
    """
    Gets called after every action made by any of the players.
    :param action: Dict containing the player uuid and the executed action
    :param round_state: Dictionary containing the round state
    """
    pass

  def receive_round_result_message(self, winners: List[Dict[str, Union[int, str]]],
                   hand_info: [List[Dict[str, Union[str, Dict]]]],
                   round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
    """
    Called at the end of the round.
    :param winners: List of the round winners containing the stack and player information.
    :param hand_info: List containing a Dict for every player at the table describing the players hand this round.
    :param round_state: Dictionary containing the round state
    """
    # round_count = round_state['round_count']
    # path = os.path.join("testcases", str(round_count)+".pickle")
    # if not os.path.exists(path):
    #   selected_round_state = {
    #     "action_histories": round_state['action_histories'],
    #     "community_cards": round_state['community_card'],
    #     "seats": round_state['seats'],
    #     "small_blind_pos": round_state['small_blind_pos'],
    #     "pot": round_state['pot']
    #   }
    #   print(json.dumps(round_state['action_histories'], sort_keys=True, indent=2))
    #   with open(path, 'wb') as f:
    #     pickle.dump(selected_round_state, f)
    # pass
