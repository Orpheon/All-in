from typing import Dict, List, Union, Tuple
from agent.random.RandomAgent import RandomAgent

import json
import pickle
import os

class LeagueTestBot(RandomAgent):
  def __init__(self):
    self.decks = []
    self.actions = []
    self.game_actions = []
    self.gains = []
    os.makedirs("testcases", exist_ok=True)

  def __str__(self):
    return "LeagueTestBot"

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
    round_count = round_state['round_count']
    path = os.path.join("testcases", str(round_count)+".pickle")
    if not os.path.exists(path):
      selected_round_state = {
        "action_histories": round_state['action_histories'],
        "community_cards": round_state['community_card'],
        "seats": round_state['seats'],
        "small_blind_pos": round_state['small_blind_pos'],
        "pot": round_state['pot']
      }
      with open(path, 'wb') as f:
        pickle.dump(selected_round_state, f)
    pass
