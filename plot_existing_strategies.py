import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from strategy.traditional_metrics import compute_playstyle, tightness_border

sns.set_style('whitegrid')

strategy_path = os.path.join("strategy", "strat_vectors")
image_path = os.path.join("strategy", "images")
strategy_files = os.listdir(strategy_path)

data = pd.DataFrame()

for file in strategy_files:
  strategy = np.load(os.path.join(strategy_path, file))
  agent_id = file.replace("_strategy.npy", "")
  agent_type = agent_id.split(" ")[0]
  print(agent_id)
  aggression, tightness = compute_playstyle(strategy)
  data = data.append({
    'agent': agent_id,
    'tightness': tightness,
    'raw_aggression': aggression,
    'normed_aggression': aggression / (1 + aggression) if np.isfinite(aggression) else 1,
    'type': agent_type
  }, ignore_index=True)

print(data)

plt.figure(figsize=(8, 8))
palette = sns.color_palette("husl", data.type.nunique())
ax = sns.scatterplot(data=data, x='tightness', y='normed_aggression', hue='type', palette=palette)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.title("Aggression vs Tightness, all agents")
os.makedirs(image_path, exist_ok=True)
ax.axhline(y=0.5, color='k')
ax.axvline(x=tightness_border(), color='k')
plt.savefig(os.path.join(image_path, "traditional_scatterplot.png"), bbox_inches='tight')