from league.agentManager import AgentManager
from league.divisionManager import DivisionManager

import pandas as pd
import numpy as np

def print_to_file(filename, dataframe):
  with open('{}.txt'.format(filename), 'w') as f:
    f.write(dataframe.to_latex(longtable=True, index=True))

if __name__ == '__main__':
  agent_manager = AgentManager(file_path='./savefiles/agent_manager.json',
                               models_path=None,
                               possible_agent_names=None)
  agent_manager.load()

  divi_manager = DivisionManager(file_path='./savefiles/divi_manager.json',
                                 divis_path='./savefiles/divis',
                                 leaderboards_path='./savefiles/leaderboards')
  divi_manager.load()

  relevant_ids = ['34', '83'] + ['47'] + ['61', '46', '27', '58', '44'] + ['39']

  divis = divi_manager.get_divi_instances(divi_ids=relevant_ids,
                                          game_engine=None,
                                          agent_manager=agent_manager)
  for _, (d, lb) in divis.items():
    d.load()
    lb[0].load()

  ts1 = divis['34'][1][0]
  ts2 = divis['83'][1][0]
  win = divis['47'][1][0]

  pctl = 20
  headers = ['name', 'type', 'origin_divi', 'generation', 'TS1', 'TS2', 'TS', 'mean', 'median', '20pctl']
  agent_table = pd.DataFrame(columns=headers)
  for agent_id, agent_info in agent_manager.agents.items():
    if not agent_info.TRAINABLE:
      divi_baselines = divis[agent_info.ORIGIN_DIVI][0].state['teachers'][:2]
      divi_clones = divis[agent_info.ORIGIN_DIVI][0].state['teachers'][2:]
      agent_table = agent_table.append({'name': agent_info.AGENT_NAME,
                                        'type': agent_info.AGENT_TYPE,
                                        'origin_divi': agent_info.ORIGIN_DIVI,
                                        'generation': 0 if agent_id in divi_baselines else divi_clones.index(agent_id) + 1 if len(divi_clones) == 10 else divi_clones.index(agent_id)//4 + 1,
                                        'TS1': ts1._ratings['current'][agent_id][0],
                                        'TS2': ts2._ratings['current'][agent_id][0],
                                        'TS': (ts1._ratings['current'][agent_id][0] + ts2._ratings['current'][agent_id][0]) / 2,
                                        'mean': sum(win._ratings['current'][agent_id].values()) / len(win._ratings['current'][agent_id]),
                                        'median': np.median(list(win._ratings['current'][agent_id].values())),
                                        '20pctl': np.percentile(list(win._ratings['current'][agent_id].values()), pctl)},
                                       ignore_index=True)

  # all printing and filtering
  sorted_by_trueskill1 = agent_table.sort_values(by=['TS'])

  print_to_file('sorted_by_trueskill', sorted_by_trueskill1)
