import random
from operator import itemgetter

import numpy as np

from agent.random.randomAgentNP import RandomAgentNP
from agent.sac1.sac1AgentNP import Sac1AgentNP
from agent.sac2.sac2AgentNP import Sac2AgentNP
from agent.qlearn1.qlearn1AgentNP import Qlearn1AgentNP
from agent.qlearn2.qlearn2AgentNP import Qlearn2AgentNP
from agent.qlearn3.qlearn3AgentNP import Qlearn3AgentNP
from agent.qlearnX.qlearnXAgentNP import QlearnXAgentNP
from agent.call.callAgentNP import CallAgentNP

from league.rating import Rating
from league.game import GameEngine
from league.logger import GenericLogger


def pick_with_probability(n, elems, probs):
  picks = []
  assert (n <= len(elems))
  for _ in range(n):
    assert (len(elems) == len(probs))
    total_prob = sum(probs)
    cumulative_probs = []
    for p in probs:
      cumulative_probs.append(p / total_prob + sum(cumulative_probs[-1:]))
    assert (len(cumulative_probs) == len(elems))
    r = random.random()
    for i, p in enumerate(cumulative_probs):
      if p >= r:
        idx = i
        break
    picks.append(elems[idx])
    del elems[idx]
    del probs[idx]
  return picks


def generate_baseline_matchup(all_agent_types):
  available_matchup_infos = [(a_class, a_info) for a_class in all_agent_types for a_info in a_class.available_agents()]

  available_learners = [(a_class, a_info) for a_class, a_info in available_matchup_infos
                        if a_info[1]['type'] == 'learner']

  # random select learners
  learners = random.sample(available_learners, 1)
  call_bots = [(CallAgentNP, ('CallAgent-{}.0'.format(i), {'type': 'teacher'})) for i in range(1,3)]
  random_bots = [(RandomAgentNP, ('RandomAgent-{}.0'.format(i), {'type': 'teacher'})) for i in range(1,4)]

  classroom = [*learners, *call_bots, *random_bots]
  instantiated = [a_type.from_id(a_info[0]) for a_type, a_info in classroom]
  return instantiated


def generate_weighted_matchup(all_agent_types, rating, n_learners, n_teachers, p_only_top_teachers):
  available_matchup_infos = [(a_class, a_info) for a_class in all_agent_types for a_info in a_class.available_agents()]

  available_learners = [(a_class, a_info) for a_class, a_info in available_matchup_infos
                        if a_info[1]['type'] == 'learner']
  available_teachers = [(a_class, a_info) for a_class, a_info in available_matchup_infos
                        if a_info[1]['type'] == 'teacher']

  # random select learners
  learners = random.sample(available_learners, n_learners)

  # trueskill weighted select
  sorted_teachers = [(rating.get_rating_from_id(a_info[0])['mu'], a_class, a_info)
                     for a_class, a_info in available_teachers]

  # encourage use of new agents
  for i, st in enumerate(sorted_teachers):
    if st[0] == 100:
      sorted_teachers[i] = (1000, sorted_teachers[i][1], sorted_teachers[i][2])

  sorted_teachers.sort(key=lambda x: x[0], reverse=True)
  top_teachers = sorted_teachers[:n_teachers]
  rest_teachers = sorted_teachers[n_teachers:]

  if random.random() < p_only_top_teachers:
    n_from_top = n_teachers
    n_from_rest = 0
  else:
    n_from_top = random.randrange(0, n_teachers)
    n_from_rest = n_teachers - n_from_top

  picked_top_teachers = pick_with_probability(n_from_top, [(a_class, a_info) for _, a_class, a_info in top_teachers],
                                              [rating for rating, _, _ in top_teachers])

  picked_rest_teachers = pick_with_probability(n_from_rest, [(a_class, a_info) for _, a_class, a_info in rest_teachers],
                                               [rating for rating, _, _ in rest_teachers])

  classroom = [*picked_top_teachers, *picked_rest_teachers, *learners]
  random.shuffle(classroom)

  instantiated = [a_type.from_id(a_info[0]) for a_type, a_info in classroom]
  return instantiated

def generate_random_teacher_matchup(all_agent_types, n_players):
  available_matchup_infos = [(a_class, a_info) for a_class in all_agent_types for a_info in a_class.available_agents()]

  available_teachers = [(a_class, a_info) for a_class, a_info in available_matchup_infos
                        if a_info[1]['type'] == 'teacher']

  classroom = random.sample(available_teachers, n_players)

  instantiated = [a_type.from_id(a_info[0]) for a_type, a_info in classroom]
  return instantiated

if __name__ == '__main__':
  ALL_AGENT_TYPES = [RandomAgentNP, Sac1AgentNP, Sac2AgentNP, Qlearn1AgentNP, CallAgentNP, Qlearn2AgentNP,
                     Qlearn3AgentNP, QlearnXAgentNP]

  BATCH_SIZE = 10_000
  INITIAL_CAPITAL = 200
  SMALL_BLIND = 1
  BIG_BLIND = 2
  N_PLAYERS = 6
  P_ONLY_TOP_TEACHERS = 0.7
  P_ZERO_LEARNERS = 0.2
  N_LEARNERS = 2
  N_GAMES_UNTIL_CLONING = (len(ALL_AGENT_TYPES) - 3) * 100 / (1-P_ZERO_LEARNERS) // N_LEARNERS
  LOGGER = GenericLogger()

  rating = Rating('./league/runner_ratings.json')
  game_engine = GameEngine(BATCH_SIZE, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND, LOGGER)

  #TODO: add again
  rating.load_ratings()

  round_counter = 0

  # gameloop
  while True:
    # TODO: configure as wanted here
    # More than 1 learner at a time overloads cuda, at least with sac1
    if random.random() <= P_ZERO_LEARNERS:
      n_learners = 0
      matchup = generate_random_teacher_matchup(ALL_AGENT_TYPES, N_PLAYERS)
    else:
      n_teachers = N_PLAYERS - N_LEARNERS
      matchup = generate_weighted_matchup(ALL_AGENT_TYPES, rating, N_LEARNERS, n_teachers, P_ONLY_TOP_TEACHERS)

    #matchup = generate_baseline_matchup(ALL_AGENT_TYPES)
    
    total_winnings = game_engine.run_game(matchup)
    winnings = np.sum(total_winnings, axis=0).tolist()

    sorted_rankings = sorted(list(zip(winnings, matchup)), key=itemgetter(0), reverse=True)

    text_rankings = '\n'.join('{}: {}'.format(m, w/BATCH_SIZE) for w, m in sorted_rankings)
    print('Round {} ({} until cloning), Winnings: \n{}'.format(round_counter, N_GAMES_UNTIL_CLONING - (round_counter % N_GAMES_UNTIL_CLONING), text_rankings))

    placings = [str(m) for _, m in sorted_rankings]
    rating.update_from_placings(placings)
    rating.save_ratings()

    round_counter += 1
    if round_counter % N_GAMES_UNTIL_CLONING == 0:
      print('Spawning new clones of all learners..')
      clone_counter = N_GAMES_UNTIL_CLONING
      available_matchup_infos = [(a_class, a_info) for a_class in ALL_AGENT_TYPES
                                 for a_info in a_class.available_agents()]
      all_learners = [(a_class, a_info) for a_class, a_info in available_matchup_infos
                      if a_info[1]['type'] == 'learner']
      for a_type, a_info in all_learners:
        instance = a_type.from_id(a_info[0])
        instance.initialize(1, INITIAL_CAPITAL, N_PLAYERS)
        instance.spawn_clone()
