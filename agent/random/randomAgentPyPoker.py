from agent.baseAgentLoadable import BaseAgent
import random


class RandomAgentPyPoker(BaseAgent):
  _fold_ratio = 1.0 / 3
  _call_ratio = 2 * _fold_ratio

  def __str__(self):
    return 'RandomAgent_{0}'.format(self.agent_id)

  @classmethod
  def _config_file_path(cls):
    return './agent/random/config.json'

  def declare_action(self, valid_actions, hole_card, round_state):
    choice = self.__choice_action(valid_actions)
    action = choice["action"]
    amount = choice["amount"]
    if action == "raise":
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
