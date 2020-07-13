import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json

import constants
from league.agentManager import AgentManager
import strategy.traditional_metrics as traditional_metrics
import strategy.multidimensional_scaling as multidimensional_scaling

sns.set_style('whitegrid')


image_path = os.path.join("strategy", "images")

agent_manager = AgentManager(
  file_path='./savefiles/agent_manager.json',
  models_path='./models',
  possible_agent_names=None
)
agent_manager.load()

def label_point(x, y, val, ax):
  a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
  for i, point in a.iterrows():
    ax.text(point['x'] + .02, point['y'], str(point['val']))

def draw_traditional_metrics(agent_manager, hue='type'):
  metrics = pd.DataFrame()

  # with open("leaderboard_perma_eval_similar.json", "r") as f:
  #   rankings = json.load(f)['current']

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
        'type': agent_info.AGENT_TYPE,
        'division': agent_info.ORIGIN_DIVI
        # 'rank': rankings[agent_id][0]
      }, ignore_index=True)

  plt.figure(figsize=(8, 8))
  if hue == 'type' or hue == 'division':
    palette = sns.color_palette("husl", metrics[hue].nunique())
    ax = sns.scatterplot(data=metrics, x='tightness', y='normed_aggression', hue=hue, palette=palette)
  elif hue == 'rank':
    ax = sns.scatterplot(data=metrics, x='tightness', y='normed_aggression', hue=hue)

  plt.xlim(-0.05, 1.05)
  plt.ylim(-0.05, 1.05)
  plt.xlabel("Tightness")
  plt.ylabel("Normed Aggression")
  plt.title("Aggression vs Tightness, all agents")
  # label_point(metrics.tightness, metrics.normed_aggression, metrics.agent, ax)
  os.makedirs(image_path, exist_ok=True)
  ax.axhline(y=0.5, color='k')
  ax.axvline(x=traditional_metrics.tightness_border(), color='k')
  plt.savefig(os.path.join(image_path, "traditional_scatterplot_"+hue+".png"), bbox_inches='tight')
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
    palette = sns.color_palette("husl", metrics['cluster'].nunique())
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
  palette = sns.color_palette("husl", data.type.nunique())
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
  palette = sns.color_palette("husl", len(agent_types))
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
  palette = sns.color_palette("husl", len(agent_types))
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

# draw_traditional_metrics(agent_manager, hue='type')
# draw_traditional_metrics(agent_manager, hue='division')
# draw_traditional_metrics(agent_manager, hue='rank', filename="traditional_scatterplot_rank.png")
# draw_mds(agent_manager)
# draw_aggression_vs_cardrank(agent_manager)
# draw_raise_size_vs_cardrank(agent_manager)
draw_traditional_metrics_kmeans(agent_manager)