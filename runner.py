from league.divisionManager import DivisionManager

from league.game import GameEngine
from league.logger import GenericLogger, NoneLogger
from league.agentManager import AgentManager


def add_all_students(agent_manager, division, origin_divi, students=('qln1', 'qln2', 'qln3', 'qln4')):
  for s in students:
    id = agent_manager.add_agent(s, True, origin_divi)
    division.state['students'].append(id)


def add_all_baselines(agent_manager, division, origin_divi, baselines=('call', 'rndm')):
  for b in baselines:
    id = agent_manager.add_agent(b, False, origin_divi)
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
  with open('league/wordlist.txt', 'r') as f:
    data = f.read()
    nouns = {d for d in data.split('\n') if len(d)}

  logger = NoneLogger()

  game_engine = GameEngine(BATCH_SIZE, INITIAL_CAPITAL, SMALL_BLIND, BIG_BLIND, logger)
  agent_manager = AgentManager(file_path='./savefiles/agent_manager.json',
                               models_path='./models',
                               possible_agent_names=nouns)

  divi_manager = DivisionManager(file_path='./savefiles/divi_manager.json',
                                 divis_path='./savefiles/divis',
                                 leaderboards_path='./savefiles/leaderboards')

  divi_manager.load()

  divi_manager.print_available_divisions()
  divi_ids_to_run = ['17', '73', '78', '25', '69', '82', '77', '34']
  divis = divi_manager.get_divi_instances(divi_ids=divi_ids_to_run,
                                          game_engine=game_engine,
                                          agent_manager=agent_manager)

  if False:  # TODO: to reset and add agents again
    add_all_baselines(agent_manager, divis['17'][0], '17')
    add_all_baselines(agent_manager, divis['73'][0], '73')
    add_all_baselines(agent_manager, divis['78'][0], '78')
    add_all_baselines(agent_manager, divis['25'][0], '25')
    add_all_baselines(agent_manager, divis['69'][0], '69')
    add_all_baselines(agent_manager, divis['82'][0], '82')
    add_all_students(agent_manager, divis['17'][0], '17', students=('qln1',))
    add_all_students(agent_manager, divis['73'][0], '73', students=('qln2',))
    add_all_students(agent_manager, divis['78'][0], '78', students=('qln1', 'qln2', 'qln4',))
    add_all_students(agent_manager, divis['25'][0], '25', students=('qln1',))
    add_all_students(agent_manager, divis['69'][0], '69', students=('qln2',))
    add_all_students(agent_manager, divis['82'][0], '82', students=('qln1', 'qln2', 'qln4',))
    for d, _ in divis.values(): d.save()
    for _, l in divis.values(): l.save()
    agent_manager.save()

  for d, _ in divis.values(): d.load()
  for _, l in divis.values(): l.load()
  agent_manager.load()

  # GAMELOOP
  round_idx = 0
  while (True):

    # run_next of divisions
    for d, _ in divis.values(): d.run_next()

    round_idx += 1
    print('[runner] round {:>5} finished \n{}'.format(round_idx, '-' * 100))

    if (round_idx % N_ROUNDS_TO_UPDATE_LEADERBOARD) == 0:
      for _, l in divis.values(): l.save()

    if (round_idx % N_ROUNDS_TO_CLONE) == 0:
      for d, _ in divis.values(): d.clone_mutables()
      agent_manager.save()
      for d, _ in divis.values(): d.save()
