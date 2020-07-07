import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import strategy.traditional_metrics as traditional_metrics
import strategy.multidimensional_scaling as multidimensional_scaling

sns.set_style('whitegrid')

strategy_path = os.path.join("strategy", "strat_vectors")
image_path = os.path.join("strategy", "images")
strategy_files = os.listdir(strategy_path)

def label_point(x, y, val, ax):
  a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
  for i, point in a.iterrows():
    ax.text(point['x'] + .02, point['y'], str(point['val']))

def load_strategies():
  data = pd.DataFrame()
  strategies = {}

  for file in strategy_files:
    strategy = np.load(os.path.join(strategy_path, file))
    agent_id = file.replace("_strategy.npy", "")
    strategies[agent_id] = strategy
    agent_type = agent_id.split(" ")[0]
    aggression, tightness = traditional_metrics.compute_playstyle(strategy)
    data = data.append({
      'agent': agent_id,
      'tightness': tightness,
      'raw_aggression': aggression,
      'normed_aggression': aggression / (1 + aggression) if np.isfinite(aggression) else 1,
      'type': agent_type
    }, ignore_index=True)

  return data, strategies

def draw_traditional_metrics(trad_metrics):
  print(trad_metrics)

  plt.figure(figsize=(8, 8))
  palette = sns.color_palette("husl", trad_metrics.type.nunique())
  ax = sns.scatterplot(data=trad_metrics, x='tightness', y='normed_aggression', hue='type', palette=palette)
  plt.xlim(-0.05, 1.05)
  plt.ylim(-0.05, 1.05)
  plt.title("Aggression vs Tightness, all agents")
  # label_point(trad_metrics.tightness, trad_metrics.normed_aggression, trad_metrics.agent, ax)
  os.makedirs(image_path, exist_ok=True)
  ax.axhline(y=0.5, color='k')
  ax.axvline(x=traditional_metrics.tightness_border(), color='k')
  plt.savefig(os.path.join(image_path, "traditional_scatterplot.png"), bbox_inches='tight')
  plt.clf()


def draw_mds(strategies):
  embeddings = multidimensional_scaling.compute_mds(strategies)
  data = pd.DataFrame()
  for k,v in embeddings.items():
    data = data.append({
      'agent': k,
      'x': v[0],
      'y': v[1],
      'type': k.split(" ")[0]
    }, ignore_index=True)

  plt.figure(figsize=(8, 8))
  palette = sns.color_palette("husl", data.type.nunique())
  ax = sns.scatterplot(data=data, x='x', y='y', hue='type', palette=palette)
  plt.title("Multidimensional scaling, all agents")
  # label_point(data.x, data.y, data.agent, ax)
  os.makedirs(image_path, exist_ok=True)
  plt.savefig(os.path.join(image_path, "mds.png"), bbox_inches='tight')
  plt.clf()

trad_metrics, strategies = load_strategies()
draw_traditional_metrics(trad_metrics)
draw_mds(strategies)