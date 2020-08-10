from league.agentManager import AgentManager
from league.divisionManager import DivisionManager

import pandas as pd
import numpy as np


def print_to_file(filename, dataframe, index):
  with open('latex_output/{}.txt'.format(filename), 'w') as f:
    f.write(dataframe.to_latex(longtable=True, index=index))


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
  headers = ['Name', 'Type', 'Division', 'Generation', 'TrueSkill1', 'TrueSkill2', 'TrueSkill', 'Mean', 'Median',
             '20-Percentile']
  agent_table = pd.DataFrame(columns=headers)

  # check unique names
  agent_names = [a_info.AGENT_NAME for _, a_info in agent_manager.agents.items()]
  print(len(agent_names), len(set(agent_names)))

  for agent_id, agent_info in agent_manager.agents.items():
    if not agent_info.TRAINABLE:
      divi_baselines = divis[agent_info.ORIGIN_DIVI][0].state['teachers'][:2]
      divi_clones = divis[agent_info.ORIGIN_DIVI][0].state['teachers'][2:]
      agent_table = agent_table.append({'Name': agent_info.AGENT_NAME,
                                        'Type': agent_info.AGENT_TYPE,
                                        'Division': agent_info.ORIGIN_DIVI,
                                        'Generation': 0 if agent_id in divi_baselines else divi_clones.index(
                                          agent_id) + 1 if len(divi_clones) == 10 else divi_clones.index(
                                          agent_id) // 4 + 1,
                                        'TrueSkill1': ts1._ratings['current'][agent_id][0],
                                        'TrueSkill2': ts2._ratings['current'][agent_id][0],
                                        'TrueSkill': (ts1._ratings['current'][agent_id][0] +
                                                      ts2._ratings['current'][agent_id][0]) / 2,
                                        'Mean': sum(win._ratings['current'][agent_id].values()) / len(
                                          win._ratings['current'][agent_id]),
                                        'Median': np.median(list(win._ratings['current'][agent_id].values())),
                                        '20-Percentile': np.percentile(list(win._ratings['current'][agent_id].values()),
                                                                       pctl)},
                                       ignore_index=True)

  print(agent_table.dtypes)

  # all printing and filtering
  fig1_sorts = ['TrueSkill', 'Mean', 'Median', '20-Percentile']
  for fig1_s in fig1_sorts:
    fig1_sorted_table = agent_table \
      .round(2) \
      .sort_values(by=[fig1_s], ascending=False) \
      .filter(items=['Name', 'TrueSkill', 'Mean', 'Median', '20-Percentile']) \
      .head(10)
    print_to_file('sorted_by_{}'.format(fig1_s), fig1_sorted_table, index=False)

  length = 20
  fig2_sorts = ['TrueSkill', 'Mean', 'Median', '20-Percentile']
  fig2_sorted_tables = [
    agent_table \
      .filter(items=['Type', 'TrueSkill', 'Mean', 'Median', '20-Percentile']) \
      .sort_values(by=[fig2_s], ascending=False)
      .filter(items=['Type'])
      .head(length)\
      .to_numpy()
    for fig2_s in fig2_sorts
  ]
  fig2_sorted_tables.insert(0, np.arange(1, length+1).reshape((length, 1)))
  print(fig2_sorted_tables)
  fig2_concatenated_tables = pd.DataFrame(np.concatenate(fig2_sorted_tables, axis=1), columns=fig2_sorts)
  print_to_file('top_agent_types_by_metric', fig2_concatenated_tables, index=True)