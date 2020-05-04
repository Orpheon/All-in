from agent.baseAgentLoadable import BaseAgent


class CallAgent(BaseAgent):

  def __str__(self):
    return 'CallAgent_{0}'.format(self.agent_id)

  @classmethod
  def _config_file_path(cls):
    return './agent/call/config.json'

  def declare_action(self, valid_actions, hole_card, round_state):
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
