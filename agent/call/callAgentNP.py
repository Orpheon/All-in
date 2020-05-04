from agent.baseAgentNP import BaseAgentNP


class CallAgentNP(BaseAgentNP):

  def __str__(self):
    return 'CallAgent_{0}'.format(self.agent_id)

  @classmethod
  def _config_file_path(cls):
    return './agent/call/config.json'

  def declare_action(self, valid_actions, hole_card, round_state):
    call_action_info = valid_actions[1]
    action, amount = call_action_info["action"], call_action_info["amount"]
    return action, amount

  def act(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser, hole_cards, community_cards):
    actions = 1
    amounts = 0
    return actions, amounts
