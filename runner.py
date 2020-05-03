import random

from configuration.CashGameConfig import CashGameConfig

from agent.allnothing.AllNothingAgent import AllNothingAgent
from agent.call.CallAgent import CallAgent
from agent.random.RandomAgentPyPoker import RandomAgent

from league.Rating import Rating


class Runner:
  _rating = Rating('./runner_placings.json')

  def __init__(self, all_agent_types, ):
    self._all_agent_types = all_agent_types

  def run_random_games(self, matchups, hands):

    for i in range(matchups):
      print('progress {0}/{1}'.format(i, matchups))

      poker_config = CashGameConfig(evaluations=hands)
      agents = self.generate_matchup()
      for agent in agents:
        poker_config.register_player(name=str(agent), algorithm=agent)
      placings = poker_config.run_evaluation()
      self._rating.update_from_placings(placings)

  def generate_matchup(self):
    available_agents = [(a_type, a_id) for a_type in self._all_agent_types for a_id in a_type.available_agent_ids()]
    picked_agents = random.sample(available_agents, 6)
    return [a_type.from_id(a_id) for a_type, a_id in picked_agents]


if __name__ == '__main__':
  all_agent_types = [CallAgent, RandomAgent, AllNothingAgent]
  different_matches = 5

  runner = Runner(all_agent_types)
  runner.run_random_games(matchups=5, hands=10_000)
  runner._rating.plot_history()
