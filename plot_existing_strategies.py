import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json

import constants
from league.agentManager import AgentManager
from league.divisionManager import DivisionManager
import strategy.traditional_metrics as traditional_metrics
import strategy.multidimensional_scaling as multidimensional_scaling

sns.set_style('whitegrid')

DIVI_NAME_TRANSLATION = {'61': 'QlnA-Cl',
                         '27': 'Qln8-Cl',
                         '46': 'SacH-Cl',
                         '58': 'SacL-Cl',
                         '44': 'AllAg-Cl',
                         '39': 'QlnA-Rn'}

image_path = os.path.join("strategy", "images")

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

def get_palette(n):
  palette = [(60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (220, 190, 255), (128, 0, 0), (0, 0, 128),
             (128, 128, 128), (0, 0, 0), (64, 64, 64)]
  palette = [(r / 255, g / 255, b / 255) for r, g, b in palette]
  return palette[:n]

def label_point(x, y, val, ax):
  a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
  for i, point in a.iterrows():
    ax.text(point['x'] + .02, point['y'], str(point['val']))

def draw_traditional_metrics(agent_manager, hue='Type', division=None):
  metrics = pd.DataFrame()

  # with open("leaderboard_perma_eval_similar.json", "r") as f:
  #   rankings = json.load(f)['current']

  for agent_id, agent_info in agent_manager.get_all_agents():
    if not agent_info.TRAINABLE and (division is None or agent_info.ORIGIN_DIVI == division):
      strategy = agent_manager.get_strategy_vector(agent_id)
      aggression = traditional_metrics.compute_aggression(strategy)
      tightness = traditional_metrics.compute_tightness(strategy)
      metrics = metrics.append({
        'agent': agent_info.AGENT_NAME + "-" + agent_id,
        'tightness': tightness,
        'normed_aggression': aggression / (1 + aggression) if np.isfinite(aggression) else 1,
        'aggression': aggression,
        'Type': agent_info.AGENT_TYPE,
        'Division': DIVI_NAME_TRANSLATION[agent_info.ORIGIN_DIVI]
        # 'rank': rankings[agent_id][0]
      }, ignore_index=True)

  plt.figure(figsize=(8, 8))
  if hue == 'Type' or hue == 'Division':
    palette = get_palette(metrics[hue].nunique())
    ax = sns.scatterplot(data=metrics, x='tightness', y='normed_aggression', hue=hue, palette=palette)
  elif hue == 'rank':
    ax = sns.scatterplot(data=metrics, x='tightness', y='normed_aggression', hue=hue)

  plt.xlim(-0.05, 1.05)
  plt.ylim(-0.05, 1.05)
  plt.xlabel("Tightness")
  plt.ylabel("Distorted Aggression")
  plt.legend(framealpha=1.0)
  # if division is None:
  #   plt.title("Aggression vs Tightness, all agents")
  # else:
  #   plt.title("Aggression vs Tightness, agents from division "+division)
  # label_point(metrics.tightness, metrics.normed_aggression, metrics.agent, ax)
  os.makedirs(image_path, exist_ok=True)
  ax.axhline(y=0.5, color='k')
  ax.axvline(x=traditional_metrics.tightness_border(), color='k')
  division_string = "" if division is None else "_"+division
  plt.savefig(os.path.join(image_path, "traditional_scatterplot_"+hue+division_string+".png"), bbox_inches='tight')
  plt.clf()

def draw_traditional_metrics_kmeans(agent_manager):
  metrics = pd.DataFrame()

  strategies = {}
  for agent_id, agent_info in agent_manager.get_all_agents():
    if not agent_info.TRAINABLE:
      strategy = agent_manager.get_strategy_vector(agent_id)
      strategies[agent_id] = strategy

  max_n_clusters = 6
  cluster_attribution = multidimensional_scaling.kmeans(strategies, MAX_N=max_n_clusters)

  for n_clusters in range(2, max_n_clusters):
    for agent_id, agent_info in agent_manager.get_all_agents():
      if not agent_info.TRAINABLE:
        strategy = agent_manager.get_strategy_vector(agent_id)
        aggression = traditional_metrics.compute_aggression(strategy)
        tightness = traditional_metrics.compute_tightness(strategy)
        metrics = metrics.append({
          'agent': agent_info.AGENT_NAME + "-" + agent_id,
          'tightness': tightness,
          'normed_aggression': aggression / (1 + aggression) if np.isfinite(aggression) else 1,
          'aggression': aggression,
          'cluster': cluster_attribution[n_clusters][agent_id]
        }, ignore_index=True)

    plt.figure(figsize=(8, 8))
    palette = get_palette(metrics['cluster'].nunique())
    ax = sns.scatterplot(data=metrics, x='tightness', y='normed_aggression', hue='cluster', palette=palette)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Tightness")
    plt.ylabel("Normed Aggression")
    plt.title("Aggression vs Tightness clustered, all agents")
    # label_point(metrics.tightness, metrics.normed_aggression, metrics.agent, ax)
    os.makedirs(image_path, exist_ok=True)
    ax.axhline(y=0.5, color='k')
    ax.axvline(x=traditional_metrics.tightness_border(), color='k')
    plt.savefig(os.path.join(image_path, "traditional_scatterplot_kmeans_"+str(n_clusters)+".png"), bbox_inches='tight')
    plt.clf()

def draw_mds(agent_manager):
  strategies = {}

  # with open("leaderboard_perma_eval_similar.json", "r") as f:
  #   rankings = json.load(f)['current']

  for agent_id, agent_info in agent_manager.get_all_agents():
    if not agent_info.TRAINABLE:
      strategy = agent_manager.get_strategy_vector(agent_id)
      strategies[agent_info.AGENT_TYPE+"-"+agent_id] = strategy

  embeddings = multidimensional_scaling.compute_mds(strategies)
  data = pd.DataFrame()
  for name,v in embeddings.items():
    data = data.append({
      'agent': name,
      'x': v[0],
      'y': v[1],
      'type': name.split("-")[0],
      'division': agent_manager.get_info(name.split("-")[1]).ORIGIN_DIVI
      # 'rank': rankings[name.split("-")[1]][0]
    }, ignore_index=True)

  plt.figure(figsize=(8, 8))
  palette = get_palette(data.type.nunique())
  ax = sns.scatterplot(data=data, x='x', y='y', hue="type", palette=palette)
  plt.title("Multidimensional scaling, all agents")
  # label_point(data.x, data.y, data.agent, ax)
  os.makedirs(image_path, exist_ok=True)
  plt.savefig(os.path.join(image_path, "mds_type.png"), bbox_inches='tight')
  plt.clf()

  # plt.figure(figsize=(8, 8))
  # ax = sns.scatterplot(data=data, x='x', y='y', hue="rank")
  # plt.title("Multidimensional scaling, all agents")
  # # label_point(data.x, data.y, data.agent, ax)
  # plt.savefig(os.path.join(image_path, "mds_rank.png"), bbox_inches='tight')
  # plt.clf()

  plt.figure(figsize=(8, 8))
  palette = sns.color_palette("husl", data.division.nunique())
  ax = sns.scatterplot(data=data, x='x', y='y', hue="division", palette=palette)
  plt.title("Multidimensional scaling, all agents")
  # label_point(data.x, data.y, data.agent, ax)
  os.makedirs(image_path, exist_ok=True)
  plt.savefig(os.path.join(image_path, "mds_division.png"), bbox_inches='tight')
  plt.clf()


def draw_aggression_vs_cardrank(agent_manager):
  agent_types = agent_manager.get_all_agent_types()
  palette = get_palette(len(agent_types))
  agent_types = {t: palette[idx] for idx, t in enumerate(agent_types)}

  # with open("leaderboard_perma_eval_similar.json", "r") as f:
  #   rankings = json.load(f)['current']

  for round_idx,round in enumerate(['preflop', 'flop', 'turn', 'river']):
    for agent_id, agent_info in agent_manager.get_all_agents():
      if not agent_info.TRAINABLE:
        column = []
        strategy = agent_manager.get_strategy_vector(agent_id)
        for card_rank in range(strategy.shape[-2]):
          aggression = traditional_metrics.compute_aggression(strategy[:, round_idx, ..., card_rank, :])
          column.append(aggression / (1 + aggression))
        sns.lineplot(
          x=np.arange(0.5, strategy.shape[-2]+0.5),
          y=column,
          color=agent_types[agent_info.AGENT_TYPE],
          legend="brief"
        )

    plt.xlim(-0.05, 10.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Card rank dectile")
    plt.ylabel("Normed Aggression")
    plt.title("Aggression by card rank in {}, all agents".format(round))
    os.makedirs(image_path, exist_ok=True)
    plt.savefig(os.path.join(image_path, "aggression_vs_rank_{}.png".format(round)), bbox_inches='tight')
    plt.clf()

def draw_raise_size_vs_cardrank(agent_manager):
  agent_types = agent_manager.get_all_agent_types()
  palette = get_palette(len(agent_types))
  agent_types = {t: palette[idx] for idx, t in enumerate(agent_types)}

  # with open("leaderboard_perma_eval_similar.json", "r") as f:
  #   rankings = json.load(f)['current']

  for round_idx, round in enumerate(['preflop', 'flop', 'turn', 'river']):
    for agent_id, agent_info in agent_manager.get_all_agents():
      if not agent_info.TRAINABLE:
        column = []
        strategy = agent_manager.get_strategy_vector(agent_id)
        for card_rank in range(strategy.shape[-2]):
          strat_condensed = multidimensional_scaling.condense(strategy[:, round_idx, ..., card_rank, :])
          amounts_raised = (strat_condensed * np.linspace(0, 200, strat_condensed.shape[-1])).sum()
          column.append(amounts_raised)
        sns.lineplot(
          x=np.arange(0.5, strategy.shape[-2] + 0.5),
          y=column,
          color=agent_types[agent_info.AGENT_TYPE],
          legend="brief"
        )

    plt.xlim(-0.05, 10.05)
    plt.ylim(-0.05, 200.05)
    plt.xlabel("Card rank dectile")
    plt.ylabel("Mean money added")
    plt.title("Money added by card rank in {}, all agents".format(round))
    os.makedirs(image_path, exist_ok=True)
    plt.savefig(os.path.join(image_path, "money_vs_rank_{}.png".format(round)), bbox_inches='tight')
    plt.clf()

draw_traditional_metrics(agent_manager, hue='Type')
# draw_traditional_metrics(agent_manager, hue='type', division="44")
draw_traditional_metrics(agent_manager, hue='Division')
# draw_mds(agent_manager)
# draw_aggression_vs_cardrank(agent_manager)
# draw_raise_size_vs_cardrank(agent_manager)
# draw_traditional_metrics_kmeans(agent_manager)