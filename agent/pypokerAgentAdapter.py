from typing import List, Dict, Union, Tuple
from pypokerengine.players import BasePokerPlayer
from agent.baseAgentNP import BaseAgentNP
import treys
import numpy as np

rounds_dict = {'preflop': 0, 'flop': 1, 'turn':2, 'river':3}
action_list = ['fold', 'call', 'raise']

class PypokerAgentAdapter(BasePokerPlayer):

  def __init__(self, agentNP):
    super().__init__()
    self._agentNP: BaseAgentNP = agentNP

  def declare_action(self, valid_actions: List[Dict[str, Union[int, str]]], hole_card: List[str],
                     round_state: Dict[str, Union[int, str, List, Dict]]) -> Tuple[Union[int, str], Union[int, str]]:
    """
    Define what action the player should execute.
    :param valid_actions: List of dictionary containing valid actions the player can execute.
    :param hole_card: Cards in possession of the player encoded as a list of strings.
    :param round_state: Dictionary containing relevant information and history of the game.
    :return: action: str specifying action type. amount: int action argument.
    """
    round_str = round_state['street']

    round = np.array([rounds_dict[round_str]])
    min_raise = np.array([valid_actions[2]['amount']['min']])
    hole_cards = np.array([[treys.card.Card.new(hc) for hc in hole_card]])
    community_cards = np.array([[treys.card.Card.new(hc) for hc in round_state['community_card']]])

    player_idx_tmp = round_state['next_player']
    folded_tmp = [p['state'] == 'folded' for p in round_state['seats']]
    current_bets_tmp = [p['stack'] for p in round_state['seats']]

    sb_idx = round_state['small_blind_pos']

    folded = np.array([folded_tmp[sb_idx:]+folded_tmp[:sb_idx]])
    player_idx = np.array([(player_idx_tmp+sb_idx) % 6])
    current_bets = np.array([current_bets_tmp[sb_idx:]+current_bets_tmp[:sb_idx]])

    seats = {p['uuid']: (idx+sb_idx) % 6 for idx, p in enumerate(round_state['seats'])}

    # find last raiser
    last_raiser_tmp = 0
    for action in reversed(round_state['action_histories'][round_str]):
      if action['action'] == 'raise':
        last_raiser_tmp = seats[action['uuid']]
        break
    last_raiser = np.array([last_raiser_tmp])

    investments = {p['uuid']: 200-p['stack'] for p in round_state['seats']}

    prev_round_investment_tmp = [0 for _ in range(6)]
    if round_str != 'preflop':
      for action in round_state['action_histories'][round_str]:
        investments[action['uuid']] -= action.get('paid', 0)
      for uuid, stack in investments.items():
        prev_round_investment_tmp[seats[uuid]] = stack

    prev_round_investment = np.array([prev_round_investment_tmp])

    actionNP, amountNP = self._agentNP.act(player_idx, round, current_bets, min_raise, prev_round_investment, folded,
                                           last_raiser, hole_cards, community_cards)
    return action_list[actionNP[0]], int(amountNP[0])


  def receive_game_start_message(self, game_info: Dict[str, Union[int, Dict, List]]) -> None:
    """
    Called once the game started.
    :param game_info: Dictionary containing game rules, # of rounds, initial stack, small blind and players at the table.
    """
    batch_size = 1
    initial_capital = 200
    n_players = 6
    self._agentNP.start_game(batch_size, initial_capital, n_players)

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
    pass
