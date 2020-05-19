import random
from operator import itemgetter

import numpy as np

from agent.random.randomAgentNP import RandomAgentNP
from agent.sac1.sac1AgentNP import Sac1AgentNP
from agent.call.callAgentNP import CallAgentNP

from league.rating import Rating
from league.game import GameEngine
from league.logger import GenericLogger


def pick_with_probability(n, elems, probs):
  picks = []
  for _ in range(n):
    total_prob = sum(probs)
    stacked_probs = []
    for p in probs:
      stacked_probs.append(p / total_prob + sum(stacked_probs[-1:]))
    r = random.random()
    idx = sum([i < r for i in stacked_probs])
    picks.append(elems[idx])
    del elems[idx]
    del probs[idx]
  return picks


def generate_matchup(all_agent_types, rating, no_learners, no_teachers):
  available_matchup_infos = [(a_class, a_info) for a_class in all_agent_types for a_info in a_class.available_agents()]
  available_learners = [(a_class, a_info) for a_class, a_info in available_matchup_infos if
                        a_info[1]['type'] == 'learner']
  available_teachers = [(a_class, a_info) for a_class, a_info in available_matchup_infos if
                        a_info[1]['type'] == 'teacher']

  print('{} learners'.format(len(available_learners)))
  print('{} teachers'.format(len(available_teachers)))

  # random select learners
  learners = random.sample(available_learners, no_learners)
  print(learners)

  # trueskill weighted select
  ratings = [rating.get_rating_from_id(a_info[0])['mu'] for _, a_info in available_teachers]
  teachers = pick_with_probability(no_teachers, available_teachers, ratings)

  classroom = [*teachers, *learners]
  random.shuffle(classroom)

  instantiated = [a_type.from_id(a_info[0]) for a_type, a_info in classroom]
  return instantiated


if __name__ == '__main__':
  all_agent_types = [RandomAgentNP, Sac1AgentNP, CallAgentNP]

  BATCH_SIZE = 10000
  INITIAL_CAPITAL = 200
  SMALL_BLIND = 1
  BIG_BLIND = 2
  NO_PLAYERS = 6
  PROBABILITY_ZERO_LEARNERS = 0.05
  LOGGER = GenericLogger()

  rating = Rating('./league/runner_ratings.json')
  game_engine = GameEngine(BATCH_SIZE, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND, LOGGER)

  # TODO: enable for persistency
  # rating.load_ratings()

  # gameloop
  while True:
    # TODO: configure as wanted here
    no_learners = random.randrange(1, NO_PLAYERS + 1)
    if random.random() <= PROBABILITY_ZERO_LEARNERS:
      no_learners = 0
    no_teachers = NO_PLAYERS - no_learners

    matchup = generate_matchup(all_agent_types, rating, no_learners, no_teachers)
    print("\nMatchup:", " ".join(str(m) for m in matchup))
    total_winnings = game_engine.run_game(matchup)
    winnings = np.sum(total_winnings, axis=0).tolist()
    print("Winnings:", " ".join(str(m) + ": " + str(w / BATCH_SIZE) for m, w in zip(matchup, winnings)))
    placings = [str(i[1]) for i in sorted(list(zip(winnings, matchup)), key=itemgetter(0), reverse=True)]
    rating.update_from_placings(placings)
    rating.save_ratings()
    #TODO: implement spawn clone
    #for agent in matchup: agent.spawn_clone()
