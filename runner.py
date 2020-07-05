import random

from league.leaderboard import Leaderboard
from league.division import RandomDivision, OverfitDivision, PermaEvalChoiceDivision, PermaEvalSampleDivision

from league.game import GameEngine
from league.logger import GenericLogger, NoneLogger
from league.agentManager import AgentManager


def add_all_students(agent_manager, division, students=('qlearn1', 'qlearn2', 'qlearn3')):
  for s in students:
    id = agent_manager.add_agent(s, True)
    division.state['students'].append(id)


def add_all_baselines(agent_manager, division, baselines=('call', 'random')):
  for b in baselines:
    id = agent_manager.add_agent(b, False)
    division.state['teachers'].append(id)


if __name__ == '__main__':
  BATCH_SIZE = 10_000
  INITIAL_CAPITAL = 200
  SMALL_BLIND = 1
  BIG_BLIND = 2
  N_PLAYERS = 6

  N_ROUNDS_TO_CLONE = 200
  N_ROUNDS_TO_UPDATE_LEADERBOARD = 50
  # logger = GenericLogger()

  #
  # basic setup
  #

  logger = NoneLogger()

  game_engine = GameEngine(BATCH_SIZE, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND, logger)
  agent_manager = AgentManager(file_path='./savefiles/agent_manager.json', models_path='./models')

  # create leaderboards
  leaderboard_random_1 = Leaderboard(file_path='./savefiles/leaderboard_random_1.json')
  leaderboard_random_2 = Leaderboard(file_path='./savefiles/leaderboard_random_2.json')
  leaderboard_random_3 = Leaderboard(file_path='./savefiles/leaderboard_random_3.json')
  leaderboard_random_4 = Leaderboard(file_path='./savefiles/leaderboard_random_4.json')
  leaderboard_random_5 = Leaderboard(file_path='./savefiles/leaderboard_random_5.json')
  leaderboard_random_6 = Leaderboard(file_path='./savefiles/leaderboard_random_6.json')
  leaderboard_perma_eval_choice = Leaderboard(file_path='./savefiles/leaderboard_perma_eval_choice.json')
  leaderboard_perma_eval_sample = Leaderboard(file_path='./savefiles/leaderboard_perma_eval_sample.json')

  all_leaderboards = [leaderboard_random_1, leaderboard_random_2, leaderboard_random_3, leaderboard_random_4,
                      leaderboard_random_5, leaderboard_random_6, leaderboard_perma_eval_choice,
                      leaderboard_perma_eval_sample]

  # create divisions
  # TODO: rename league to division
  divi_random_1 = RandomDivision('savefiles/division_random_1.json', game_engine, leaderboard_random_1, agent_manager)
  divi_random_2 = RandomDivision('savefiles/division_random_2.json', game_engine, leaderboard_random_2, agent_manager)
  divi_random_3 = RandomDivision('savefiles/division_random_3.json', game_engine, leaderboard_random_3, agent_manager)
  divi_random_4 = RandomDivision('savefiles/division_random_4.json', game_engine, leaderboard_random_4, agent_manager)
  divi_random_5 = RandomDivision('savefiles/division_random_5.json', game_engine, leaderboard_random_5, agent_manager)
  divi_random_6 = RandomDivision('savefiles/division_random_6.json', game_engine, leaderboard_random_6, agent_manager)
  divi_perma_eval_choice = PermaEvalChoiceDivision('savefiles/division_perma_eval_choice.json', game_engine,
                                                   leaderboard_perma_eval_choice, agent_manager)
  divi_perma_eval_sample = PermaEvalSampleDivision('savefiles/division_perma_eval_sample.json', game_engine,
                                                   leaderboard_perma_eval_sample, agent_manager)

  all_divi = [divi_random_1, divi_random_2, divi_random_3, divi_random_4, divi_random_5, divi_random_6,
              divi_perma_eval_choice, divi_perma_eval_sample]

  # load / add agents
  agent_manager.load()

  # add_all_students(agent_manager, division_random_1, students=('qlearn1',))
  # add_all_students(agent_manager, division_random_2, students=('qlearn2',))
  # add_all_students(agent_manager, division_random_3, students=('qlearn3',))
  # add_all_students(agent_manager, division_random_4, students=('qlearn1', 'qlearn2'))
  # add_all_students(agent_manager, division_random_5, students=('qlearn1', 'qlearn2', 'qlearn3'))
  # add_all_students(agent_manager, division_random_6, students=('qlearn1', 'qlearn2', 'qlearn3'))

  # add_all_baselines(agent_manager, division_random_1)
  # add_all_baselines(agent_manager, division_random_2)
  # add_all_baselines(agent_manager, division_random_3)
  # add_all_baselines(agent_manager, division_random_4)
  # add_all_baselines(agent_manager, division_random_5)
  # division_random_6.state['teachers'].append('36536334')
  # agent_manager._add_agent_with_id('36536334', 'qlearn1', False)

  # load leaderboards
  for l in all_leaderboards: l.load()

  # load divisions
  for d in all_divi: d.load()

  # agent_manager.save()

  # GAMELOOP
  round_idx = 0
  while (True):

    # run_next of divisions
    for d in all_divi: d.run_next()

    round_idx += 1
    print('[runner] round {:>5} finished \n{}'.format(round_idx, '-' * 100))

    if (round_idx % N_ROUNDS_TO_UPDATE_LEADERBOARD) == 0:
      # save leaderboards realtime
      for l in all_leaderboards: l.save()

    if (round_idx % N_ROUNDS_TO_CLONE) == 0:
      # call clone
      divi_random_1.clone_mutables()
      divi_random_2.clone_mutables()
      divi_random_3.clone_mutables()
      divi_random_4.clone_mutables()
      divi_random_5.clone_mutables()
      divi_random_6.clone_mutables()

      # save agent manager
      agent_manager.save()

      # save divisions
      for d in all_divi: d.save()
