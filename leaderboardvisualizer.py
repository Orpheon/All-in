from league.agentManager import AgentManager
from league.divisionManager import DivisionManager

if __name__ == '__main__':
  agent_manager = AgentManager(file_path='./savefiles/agent_manager.json',
                               models_path=None,
                               possible_agent_names=None)
  agent_manager.load()

  divi_manager = DivisionManager(file_path='./savefiles/divi_manager.json',
                                 divis_path='./savefiles/divis',
                                 leaderboards_path='./savefiles/leaderboards')
  divi_manager.load()

  relevant_ids = ['28'] + ['72', '75'] + ['65', '74', '37', '42']

  divis = divi_manager.get_divi_instances(divi_ids=relevant_ids,
                                          game_engine=None,
                                          agent_manager=agent_manager)
  for _, ls in divis.values():
    for l in ls:
      l.load()
      l.print_leaderboard(agent_manager)
      l.plot_leaderboard(agent_manager)
