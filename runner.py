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


def generate_matchup(all_agent_types, rating, n_learners, n_teachers):
  available_matchup_infos = [(a_class, a_info) for a_class in all_agent_types for a_info in a_class.available_agents()]
  available_learners = [(a_class, a_info) for a_class, a_info in available_matchup_infos if
                        a_info[1]['type'] == 'learner']
  available_teachers = [(a_class, a_info) for a_class, a_info in available_matchup_infos if
                        a_info[1]['type'] == 'teacher']

  # random select learners
  learners = random.sample(available_learners, n_learners)

  # trueskill weighted select
  ratings = [rating.get_rating_from_id(a_info[0])['mu']**2 for _, a_info in available_teachers]
  teachers = pick_with_probability(n_teachers, available_teachers, ratings)

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
  N_PLAYERS = 6
  PROBABILITY_ZERO_LEARNERS = 0.05
  N_GAMES_UNTIL_CLONING = 500
  LOGGER = GenericLogger()

  rating = Rating('./league/runner_ratings.json')
  game_engine = GameEngine(BATCH_SIZE, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND, LOGGER)

  # TODO: enable for persistency
  # rating.load_ratings()

  round_counter = 0

  # gameloop
  while True:
    # TODO: configure as wanted here
    # More than 1 learner at a time overloads cuda, at least with sac1
    n_learners = 1
    if random.random() <= PROBABILITY_ZERO_LEARNERS:
      n_learners = 0
    n_teachers = N_PLAYERS - n_learners

    matchup = generate_matchup(all_agent_types, rating, n_learners, n_teachers)
    total_winnings = game_engine.run_game(matchup)
    winnings = np.sum(total_winnings, axis=0).tolist()
    print("Round {0}, Winnings: {1}".format(round_counter, " ".join(str(m) + ": " + str(w / BATCH_SIZE) for m, w in zip(matchup, winnings))))
    placings = [str(i[1]) for i in sorted(list(zip(winnings, matchup)), key=itemgetter(0), reverse=True)]
    rating.update_from_placings(placings)
    rating.save_ratings()

    round_counter += 1
    if round_counter % N_GAMES_UNTIL_CLONING == 0:
      print("Spawning new clones of all learners..")
      clone_counter = N_GAMES_UNTIL_CLONING
      available_matchup_infos = [(a_class, a_info) for a_class in all_agent_types for a_info in
                                 a_class.available_agents()]
      all_learners = [(a_class, a_info) for a_class, a_info in available_matchup_infos if
                        a_info[1]['type'] == 'learner']
      for a_type, a_info in all_learners:
        instance = a_type.from_id(a_info[0])
        instance.initialize(1, INITIAL_CAPITAL, N_PLAYERS)
        instance.spawn_clone()