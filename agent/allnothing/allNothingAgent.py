from agent.baseAgentLoadable import BaseAgent


class AllNothingAgent(BaseAgent):

  ev_table = {'AA': 2.32, 'KK': 1.67, 'QQ': 1.22, 'JJ': 0.86, 'AK s': 0.78, 'AQ s': 0.59, 'TT': 0.58, 'AK': 0.51,
              'AJ s': 0.44, 'KQ s': 0.39, '99': 0.38, 'AT s': 0.32, 'AQ': 0.31, 'KJ s': 0.29, '88': 0.25,
              'QJ s': 0.23, 'KT s': 0.2, 'A9 s': 0.19, 'AJ': 0.19, 'QT s': 0.17, 'KQ': 0.16, '77': 0.16, 'JT s': 0.15,
              'A8 s': 0.1, 'K9 s': 0.09, 'AT': 0.08, 'A5 s': 0.08, 'A7 s': 0.0, 'KJ': 0.08, '66': 0.07, 'T9 s': 0.05,
              'A4 s': 0.05, 'Q9 s': 0.05, 'J9 s': 0.04, 'QJ': 0.03, 'A6 s': 0.03, '55': 0.02, 'A3 s': 0.02,
              'K8 s': 0.01, 'KT': 0.01, '98 s': 0.0, 'T8 s': -0.0, 'K7 s': -0.0, 'A2 s': 0.0}

  def __str__(self):
    return 'AllNothingAgent_{0}'.format(self.agent_id)

  @classmethod
  def _config_file_path(cls):
    return './agent/allnothing/config.json'

  def declare_action(self, valid_actions, hole_card, round_state):
    if round_state['street'] == 'preflop':
      cards = ''.join([i[0] for i in hole_card])
      same_suit = 's' if hole_card[0][1] == hole_card[1][1] else ''
      card_lbl = '{0} {1}'.format(cards, same_suit)
      ev = self.ev_table.get(card_lbl, -1)
      if ev > self.config['min_ev_all_in']:
        return 'raise', valid_actions[2]['amount']['max']
      if ev < self.config['max_ev_fold']:
        return 'fold', 0
    call_action_info = valid_actions[1]
    action, amount = call_action_info["action"], call_action_info["amount"]
    return action, amount

  def receive_game_start_message(self, game_info):
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, new_action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass
