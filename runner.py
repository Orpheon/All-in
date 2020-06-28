import random

from league.leaderboard import Leaderboard
from league.division import NormalDivision

from league.game import GameEngine
from league.logger import GenericLogger, NoneLogger
from league.agentManager import AgentManager

if __name__ == '__main__':
  BATCH_SIZE = 10_000
  INITIAL_CAPITAL = 200
  SMALL_BLIND = 1
  BIG_BLIND = 2
  N_PLAYERS = 6

  N_ROUNDS_TO_CLONE = 100

  # logger = GenericLogger()
  logger = NoneLogger()

  game_engine = GameEngine(BATCH_SIZE, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND, logger)

  agent_manager = AgentManager(file_path='./savefiles/agent_manager.json', models_path='./models')
  leaderboard_1 = Leaderboard(file_path='./savefiles/leaderboard_1.json')
  division_1 = NormalDivision('./savefiles/league_one.json', game_engine, leaderboard_1, agent_manager)

  # add_all_agents(agent_manager)

  agent_manager.load()
  leaderboard_1.load()
  division_1.load()

  round_idx = 0
  while (True):
    for _ in range(N_ROUNDS_TO_CLONE):
      division_1.run_next()
      round_idx += 1
      print('[runner] round {:>5} finished \n{}'.format(round_idx, '-' * 100))
    division_1.clone_mutables()
    agent_manager.save()
    leaderboard_1.save()
    division_1.save()


def add_all_agents(agent_manager):
  # add agents
  students = ['qlearn1', 'qlearn2', 'qlearn3']
  for s in students:
    id = agent_manager.add_agent(s, True)
    division_1.agents['students'].append(id)

  teachers = ['call', 'random']
  for t in teachers:
    id = agent_manager.add_agent(t, False)
    division_1.agents['teachers'].append(id)


#
# TODO: remove
#

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
                        if a_info[1]['matchup_info']['type'] == 'learner']

  # random select learners
  learners = random.sample(available_learners, 1)

  # set baselines
  call_bots = [(CallAgentNP, ('CallAgent-{}.0'.format(i), {'type': 'teacher'})) for i in range(1, 3)]
  random_bots = [(RandomAgentNP, ('RandomAgent-{}.0'.format(i), {'type': 'teacher'})) for i in range(1, 4)]

  classroom = [*learners, *call_bots, *random_bots]
  instantiated = [a_type.from_id(a_info[0]) for a_type, a_info in classroom]
  return instantiated


def generate_weighted_matchup(all_agent_types, rating, n_learners, n_teachers, p_only_top_teachers):
  available_matchup_infos = [(a_class, a_info) for a_class in all_agent_types for a_info in a_class.available_agents()]

  available_learners = [(a_class, a_info) for a_class, a_info in available_matchup_infos
                        if a_info[1]['matchup_info']['type'] == 'learner']
  available_teachers = [(a_class, a_info) for a_class, a_info in available_matchup_infos
                        if a_info[1]['matchup_info']['type'] == 'teacher']

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
                        if a_info[1]['matchup_info']['type'] == 'teacher']

  ages = [a_info[1]['matchup_info']['age'] for _, a_info in available_teachers]
  max_age = max(ages)
  classroom = pick_with_probability(n_players, available_teachers, [max_age - age + 1 for age in ages])

  instantiated = [a_type.from_id(a_info[0]) for a_type, a_info in classroom]
  return instantiated
