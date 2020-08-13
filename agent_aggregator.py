from league.agentManager import AgentManager
from league.divisionManager import DivisionManager
import strategy.traditional_metrics as traditional_metrics

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import os

image_path = "final_images"
sns.set_style('whitegrid')

DIVI_NAME_TRANSLATION = {'61': 'QlnA-Cl',
                         '27': 'Qln8-Cl',
                         '46': 'SacH-Cl',
                         '58': 'SacL-Cl',
                         '44': 'AllAg-Cl',
                         '39': 'QlnA-Rn'}

def print_to_file(filename, dataframe, index):
  with open('latex_output/{}.txt'.format(filename), 'w') as f:
    f.write(dataframe.to_latex(longtable=True, index=index))

def get_palette(n):
  palette = [(60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (220, 190, 255), (128, 0, 0), (0, 0, 128),
             (128, 128, 128), (0, 0, 0), (64, 64, 64)]
  palette = [(r / 255, g / 255, b / 255) for r, g, b in palette]
  return palette[:n]

def plot_rank(agent_table, rank):
  plt.figure(figsize=(8, 8))
  normed = agent_table.copy()
  normed[rank] = (normed[rank] - normed[rank].mean()) / (normed[rank].std())
  # sns.set_palette(sns.color_palette("RdBu_r"))
  cmap = sns.diverging_palette(20, 220, sep=20, as_cmap=True)
  ax = sns.scatterplot(data=normed, x='Tightness', y='Distorted Aggression', hue=rank, palette=cmap)
  plt.xlim(-0.05, 1.05)
  plt.ylim(-0.05, 1.05)
  plt.xlabel("Tightness")
  plt.ylabel("Distorted Aggression")
  ax.get_legend().remove()
  # label_point(metrics.tightness, metrics.normed_aggression, metrics.agent, ax)
  os.makedirs(image_path, exist_ok=True)
  ax.axhline(y=0.5, color='k')
  ax.axvline(x=traditional_metrics.tightness_border(), color='k')
  plt.savefig(os.path.join(image_path, "aggtightrank"+rank+".png"), bbox_inches='tight')
  plt.clf()

def plot_agent_dist(agent_table, rank):
  plt.figure(figsize=(8, 8))
  ax = sns.boxenplot(data=agent_table, x='Type', y=rank, scale="linear", palette=get_palette(agent_table['Type'].nunique()))
  # ax.get_legend().remove()
  # label_point(metrics.tightness, metrics.normed_aggression, metrics.agent, ax)
  os.makedirs(image_path, exist_ok=True)
  plt.savefig(os.path.join(image_path, "agentdist"+rank+".png"), bbox_inches='tight')
  plt.clf()

def plot_matchup_dist(agent_table, rank):
  subset = agent_table[agent_table.Division.isin(['61', '39'])]
  subset['Division'] = subset['Division'].replace('61', DIVI_NAME_TRANSLATION['61'])
  subset['Division'] = subset['Division'].replace('39', DIVI_NAME_TRANSLATION['39'])
  plt.figure(figsize=(8, 8))
  ax = sns.boxenplot(data=subset, x='Division', y=rank, scale="linear", palette=get_palette(agent_table['Type'].nunique()))
  # ax.get_legend().remove()
  # label_point(metrics.tightness, metrics.normed_aggression, metrics.agent, ax)
  os.makedirs(image_path, exist_ok=True)
  plt.savefig(os.path.join(image_path, "matchupdist"+rank+".png"), bbox_inches='tight')
  plt.clf()

def plot_scatter_trueskill(agent_table):
  sorted = agent_table.sort_values('TrueSkill1')
  ax = sns.scatterplot(data=sorted, x='TrueSkill1', y='TrueSkill2', hue='Type')
  plt.xlabel("TrueSkill Leaderboard 1")
  plt.ylabel("TrueSkill Leaderboard 2")
  # ax.get_legend().remove()
  # label_point(metrics.tightness, metrics.normed_aggression, metrics.agent, ax)
  os.makedirs(image_path, exist_ok=True)
  # ax.axhline(y=0.5, color='k')
  # ax.axvline(x=traditional_metrics.tightness_border(), color='k')
  plt.savefig(os.path.join(image_path, "trueskill_comparison.png"), bbox_inches='tight')
  plt.clf()

def plot_generations(agent_table, rank):
  combined = agent_table.copy()
  combined['Division-Type'] = combined['Readable-Division'] + "-" + combined['Type']
  plt.figure(figsize=(8, 8))
  for type in combined['Division-Type'].unique():
    if "Call" not in type and "Random" not in type:
      ax = sns.lineplot(data=combined[combined['Division-Type'] == type], x='Generation', y=rank, palette=get_palette(combined['Division-Type'].nunique()), label=type, estimator=None)
  # ax.get_legend().remove()
  # label_point(metrics.tightness, metrics.normed_aggression, metrics.agent, ax)
  os.makedirs(image_path, exist_ok=True)
  plt.savefig(os.path.join(image_path, "generations"+rank+".png"), bbox_inches='tight')
  plt.clf()

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
      strategy = agent_manager.get_strategy_vector(agent_id)
      aggression = traditional_metrics.compute_aggression(strategy)
      tightness = traditional_metrics.compute_tightness(strategy)
      agent_table = agent_table.append({'Name': agent_info.AGENT_NAME,
                                        'Type': agent_info.AGENT_TYPE,
                                        'Division': agent_info.ORIGIN_DIVI,
                                        'Readable-Division': DIVI_NAME_TRANSLATION[agent_info.ORIGIN_DIVI],
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
                                                                       pctl),
                                        'Tightness': tightness,
                                        'Distorted Aggression': aggression / (1 + aggression) if np.isfinite(
                                          aggression) else 1
                                        },
                                       ignore_index=True)


  plot_rank(agent_table, 'TrueSkill')
  plot_rank(agent_table, 'Mean')
  plot_rank(agent_table, 'Median')
  plot_rank(agent_table, '20-Percentile')
  # plot_agent_dist(agent_table, 'TrueSkill')
  # plot_agent_dist(agent_table, 'Mean')
  # plot_agent_dist(agent_table, 'Median')
  # plot_agent_dist(agent_table, '20-Percentile')
  # plot_matchup_dist(agent_table, 'TrueSkill')
  # plot_matchup_dist(agent_table, 'Mean')
  # plot_matchup_dist(agent_table, 'Median')
  # plot_matchup_dist(agent_table, '20-Percentile')
  # plot_scatter_trueskill(agent_table)
  # plot_generations(agent_table, 'TrueSkill')
  # plot_generations(agent_table, 'Mean')
  # plot_generations(agent_table, 'Median')
  # plot_generations(agent_table, '20-Percentile')
