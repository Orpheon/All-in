import random
from operator import itemgetter

import numpy as np

from agent.random.randomAgentNP import RandomAgentNP

from league.rating import Rating
from league.game import GameEngine
from league.logger import GenericLogger


def generate_matchup(all_agent_types):
  available_agents = [(a_type, a_id) for a_type in all_agent_types for a_id in a_type.available_agent_ids()]
  picked_agents = random.sample(available_agents, 6)
  return [a_type.from_id(a_id) for a_type, a_id in picked_agents]


if __name__ == '__main__':
  all_agent_types = [RandomAgentNP]

  BATCH_SIZE = 10000
  INITIAL_CAPITAL = 200
  SMALL_BLIND = 1
  BIG_BLIND = 2
  LOGGER = GenericLogger()

  rating = Rating('./runner_placings.json')
  game_engine = GameEngine(BATCH_SIZE, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND, LOGGER)

  # gameloop
  for _ in range(10):
    matchup = generate_matchup(all_agent_types)
    total_winnings = game_engine.run_game(matchup)
    winnings = np.sum(total_winnings, axis=0).tolist()
    placings = [str(i[1]) for i in sorted(list(zip(winnings, matchup)), key=itemgetter(0), reverse=True)]
    rating.update_from_placings(placings)
  rating.plot_history()
