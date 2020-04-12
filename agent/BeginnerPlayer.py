from typing import Dict, List, Union, Tuple
from pypokerengine.players import BasePokerPlayer


class BeginnerPlayer(BasePokerPlayer):
    """
    Documentation for callback arguments given here:
    https://github.com/ishikota/PyPokerEngine/blob/master/AI_CALLBACK_FORMAT.md
    """

    id = 'BeginnerPlayer'

    ev_table = {'AA': 2.32, 'KK': 1.67, 'QQ': 1.22, 'JJ': 0.86, 'AK s': 0.78, 'AQ s': 0.59, 'TT': 0.58, 'AK': 0.51,
                'AJ s': 0.44, 'KQ s': 0.39, '99': 0.38, 'AT s': 0.32, 'AQ': 0.31, 'KJ s': 0.29, '88': 0.25,
                'QJ s': 0.23, 'KT s': 0.2, 'A9 s': 0.19, 'AJ': 0.19, 'QT s': 0.17, 'KQ': 0.16, '77': 0.16, 'JT s': 0.15,
                'A8 s': 0.1, 'K9 s': 0.09, 'AT': 0.08, 'A5 s': 0.08, 'A7 s': 0.0, 'KJ': 0.08, '66': 0.07, 'T9 s': 0.05,
                'A4 s': 0.05, 'Q9 s': 0.05, 'J9 s': 0.04, 'QJ': 0.03, 'A6 s': 0.03, '55': 0.02, 'A3 s': 0.02,
                'K8 s': 0.01, 'KT': 0.01, '98 s': 0.0, 'T8 s': -0.0, 'K7 s': -0.0, 'A2 s': 0.0}

    def __init__(self, min_ev_all_in= 0, max_ev_fold=0):
        self.min_ev_all_in = min_ev_all_in
        self.max_ev_fold = max_ev_fold
        self.id = 'BeginnerPlayer[{0}][{1}]'.format(min_ev_all_in, max_ev_fold)

    def __str__(self):
        return "BeginnerPlayer"

    def declare_action(self, valid_actions: List[Dict[str, Union[int, str]]], hole_card: List[str],
                       round_state: Dict[str, Union[int, str, List, Dict]]) -> Tuple[Union[int, str], Union[int, str]]:
        """
        Define what action the player should execute.
        :param valid_actions: List of dictionary containing valid actions the player can execute.
        :param hole_card: Cards in possession of the player encoded as a list of strings.
        :param round_state: Dictionary containing relevant information and history of the game.
        :return: action: str specifying action type. amount: int action argument.
        """
        if round_state['street'] == 'preflop':
            cards = ''.join([i[0] for i in hole_card])
            same_suit = 's' if hole_card[0][1] == hole_card[1][1] else ''
            card_lbl = '{0} {1}'.format(cards, same_suit)
            ev = self.ev_table.get(card_lbl, -1)
            if ev > self.min_ev_all_in:
                return 'raise', valid_actions[2]['amount']['max']
            if ev < self.max_ev_fold:
                return 'fold', 0
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount

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
        pass