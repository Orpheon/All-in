from league.agentManager import AgentManager
from league.divisionManager import DivisionManager

import pandas as pd
import numpy as np
from scipy import stats


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

  DIVI_NAME_TRANSLATION = {'61': 'QlnA-Cl',
                           '27': 'Qln8-Cl',
                           '46': 'SacH-Cl',
                           '58': 'SacL-Cl',
                           '44': 'AllAg-Cl',
                           '39': 'QlnA-Rn'}

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
  headers = ['Agent Id', 'Name', 'Type', 'Division Id', 'Division Type', 'Generation', 'TrueSkill1', 'TrueSkill2',
             'TrueSkill', 'TrueSkillLocal', 'Mean', 'Median',
             '20-Percentile']
  agent_table = pd.DataFrame(columns=headers)

  # check unique names
  agent_names = [a_info.AGENT_NAME for _, a_info in agent_manager.agents.items()]
  print('agent names vs unique agent names', len(agent_names), len(set(agent_names)))

  for agent_id, agent_info in agent_manager.agents.items():
    if not agent_info.TRAINABLE:
      divi_baselines = divis[agent_info.ORIGIN_DIVI][0].state['teachers'][:2]
      divi_clones = divis[agent_info.ORIGIN_DIVI][0].state['teachers'][2:]
      agent_table = agent_table.append({'Agent Id': agent_id,
                                        'Name': agent_info.AGENT_NAME,
                                        'Type': agent_info.AGENT_TYPE,
                                        'Division Id': agent_info.ORIGIN_DIVI,
                                        'Division Type': DIVI_NAME_TRANSLATION[agent_info.ORIGIN_DIVI],
                                        'Generation': 0 if agent_id in divi_baselines else divi_clones.index(
                                          agent_id) + 1 if len(divi_clones) == 10 else divi_clones.index(
                                          agent_id) // 4 + 1,
                                        'TrueSkill1': ts1._ratings['current'][agent_id][0],
                                        'TrueSkill2': ts2._ratings['current'][agent_id][0],
                                        'TrueSkill': (ts1._ratings['current'][agent_id][0] +
                                                      ts2._ratings['current'][agent_id][0]) / 2,
                                        'TrueSkillLocal':
                                          divis[agent_info.ORIGIN_DIVI][1][0]._ratings['current'].get(agent_id,
                                                                                                      [np.nan])[0],
                                        'Mean': sum(win._ratings['current'][agent_id].values()) / len(
                                          win._ratings['current'][agent_id]),
                                        'Median': np.median(list(win._ratings['current'][agent_id].values())),
                                        '20-Percentile': np.percentile(list(win._ratings['current'][agent_id].values()),
                                                                       pctl)},
                                       ignore_index=True)

  # print(agent_table.filter(items=['Name', 'Division Id', 'TrueSkillLocal']))

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
    agent_table
      .filter(items=['Type', 'TrueSkill', 'Mean', 'Median', '20-Percentile'])
      .sort_values(by=[fig2_s], ascending=False)
      .filter(items=['Type'])
      .head(length)
      .to_numpy()
    for fig2_s in fig2_sorts
  ]
  fig2_sorted_tables.insert(0, np.arange(1, length + 1).reshape((length, 1)))
  fig2_concatenated_tables = pd.DataFrame(np.concatenate(fig2_sorted_tables, axis=1), columns=['Rank'] + fig2_sorts)
  print_to_file('top_agent_types_by_metric', fig2_concatenated_tables, index=False)

  length = 20
  fig3_sorts = ['TrueSkill', 'Mean', 'Median', '20-Percentile']
  fig3_sorted_tables = [
    agent_table
      .filter(items=['Division Type', 'TrueSkill', 'Mean', 'Median', '20-Percentile'])
      .sort_values(by=[fig3_s], ascending=False)
      .filter(items=['Division Type'])
      .head(length)
      .to_numpy()
    for fig3_s in fig3_sorts
  ]
  fig3_sorted_tables.insert(0, np.arange(1, length + 1).reshape((length, 1)))
  fig3_concatenated_tables = pd.DataFrame(np.concatenate(fig3_sorted_tables, axis=1), columns=['Rank'] + fig3_sorts)
  print_to_file('agent_division_according_to_metric', fig3_concatenated_tables, index=False)

  fig4_divisions = list(DIVI_NAME_TRANSLATION.values())
  fig4_sorted_divisions = {
    fig4_divi:
      stats.kendalltau(*[
        agent_table
        [agent_table['Division Type'] == fig4_divi]
                       .dropna(axis=0, )
                       .filter(items=['Agent Id', 'Division Type', board])
                       .sort_values(by=[board], ascending=False)
                       .filter(items=['Agent Id'])
                       .to_numpy()[:, 0]

        for board in ['TrueSkillLocal', 'TrueSkill']])
    for fig4_divi in fig4_divisions
  }
  fig4_table = pd.DataFrame(columns=['Division', 'Kendall’s Tau'])
  for d, v in fig4_sorted_divisions.items():
    fig4_table = fig4_table.append({'Division': d, 'Kendall’s Tau': v[0]}, ignore_index=True)
  fig4_table = fig4_table.round(3)
  print_to_file('Kendal tau between divisions', fig4_table, index=False)

  fig5_sorts = ['TrueSkill', 'Mean', 'Median', '20-Percentile']
  fig5_sorted_tables = {
    fig5_s:
      agent_table
        .filter(items=['Agent Id', 'TrueSkill', 'Mean', 'Median', '20-Percentile'])
        .sort_values(by=[fig5_s], ascending=False)
        .filter(items=['Agent Id'])
        .to_numpy()[:, 0]
    for fig5_s in fig5_sorts
  }

  fig5_win_lists = {
    fig5_metric:
      [win._ratings['current'][id_a][id_b] < 0
       for idx_a, id_a in enumerate(fig5_sorted_table)
       for idx_b, id_b in enumerate(fig5_sorted_table) if idx_a < idx_b]
    for fig5_metric, fig5_sorted_table in fig5_sorted_tables.items()
  }
  fig5_table = pd.DataFrame(columns=['Metric', 'Upsets'])
  for k,v in fig5_win_lists.items():
    fig5_table = fig5_table.append({'Metric': k, 'Upsets': sum(v)/len(v)}, ignore_index=True)
  print_to_file('upsets_per_metric', fig5_table, index=False)
