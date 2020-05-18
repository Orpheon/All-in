import random
from operator import itemgetter

import numpy as np

from agent.random.randomAgentNP import RandomAgentNP
from agent.sac1.sac1AgentNP import Sac1AgentNP
from agent.call.callAgentNP import CallAgentNP

from league.rating import Rating
from league.game import GameEngine
from league.logger import GenericLogger


def generate_matchup(all_agent_types):
  mutators = []
  comparisons = []
  available_matchup_infos = [(a_type, agent_info) for a_type in all_agent_types for agent_info in a_type.available_agents()]

  # random select
  available_ids = [(a_type, a_info[0]) for a_type, a_info in available_matchup_infos]

  picked_agents = random.sample(available_ids, 6)
  instantiated = [a_type.from_id(a_id) for a_type, a_id in picked_agents]
  return instantiated

if __name__ == '__main__':
  # all_agent_types = [RandomAgentNP, Sac1AgentNP, CallAgentNP]
  all_agent_types = [CallAgentNP, RandomAgentNP]

  BATCH_SIZE = 10000
  INITIAL_CAPITAL = 200
  SMALL_BLIND = 1
  BIG_BLIND = 2
  LOGGER = GenericLogger()

  rating = Rating('./league/runner_ratings.json')
  game_engine = GameEngine(BATCH_SIZE, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND, LOGGER)

  # gameloop

  #TODO: enable for persistency
  #rating.load_ratings()
  while True:
    matchup = generate_matchup(all_agent_types)
    print("\nMatchup:", " ".join(str(m) for m in matchup))
    total_winnings = game_engine.run_game(matchup)
    winnings = np.sum(total_winnings, axis=0).tolist()
    print("Winnings:", " ".join(str(m)+": "+str(w / BATCH_SIZE) for m, w in zip(matchup, winnings)))
    placings = [str(i[1]) for i in sorted(list(zip(winnings, matchup)), key=itemgetter(0), reverse=True)]
    rating.update_from_placings(placings)
    rating.save_ratings()
